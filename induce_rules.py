#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
induce_rules.py

Rule induction for ICCU on TOFU.

Pipeline
--------
1) Embed the questions of a TOFU forget split with an embedding model.
2) Cluster the embeddings into K clusters with k-means.
3) For each cluster, select representative questions (closest to the
   cluster centroid) and prompt an LLM to summarize them into one
   natural-language refusal rule.
4) Save one record per cluster (centroid + rule text) to a JSONL file.

Embeddings
----------
Embeddings are mean-pooled over the last hidden layer and L2-normalized.
The script first tries to load them from the embedding cache shared with
eval_tofu.py (metadata.json, questions.jsonl, embeds.npy). If the cache is
missing or does not match the current questions exactly, the embedding model
is loaded and the embeddings are computed directly. These embeddings are used
only in the current run and are not written to the cache.

Cache key format:
  dataset_name || config || split || embedding_model_id || embed_max_length || normalize
"""

import argparse
import gc
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from sklearn.cluster import KMeans
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# =========================================================
# General helpers
# =========================================================
def cleanup_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def safe_filename(s: str) -> str:
    return (
        s.replace("/", "__")
         .replace(":", "_")
         .replace(" ", "_")
         .replace("|", "_")
    )


def truncate(s: str, limit: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit] + "..."


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# Dataset
# =========================================================
def get_local_dataset_path(dataset_name: str, config: str, split: str) -> str:
    name = dataset_name.replace("/", "_")
    config = config.replace("/", "_")
    return f"./local_datasets/{name}_{config}_{split}"


def load_dataset_items(dataset_name: str, dataset_config: str, split: str):
    local_path = get_local_dataset_path(dataset_name, dataset_config, split)

    if os.path.exists(local_path):
        print(f"[LOAD LOCAL] {local_path}")
        ds = load_from_disk(local_path)
    else:
        print(f"[DOWNLOAD FROM HF] {dataset_name}/{dataset_config}/{split}")
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=False,
            download_mode="reuse_dataset_if_exists",
        )
    return ds


def build_questions(
    ds,
    max_questions: Optional[int],
) -> Tuple[List[str], List[int]]:
    """Collect non-empty question texts and their original dataset indices."""
    questions: List[str] = []
    original_indices: List[int] = []

    print(">>> Building clustering inputs (question field only) ...")
    for i, item in enumerate(ds):
        question_text = str(item.get("question", "") or "").strip()
        if not question_text:
            continue

        questions.append(question_text)
        original_indices.append(i)

        if max_questions is not None and len(questions) >= max_questions:
            break

    return questions, original_indices


# =========================================================
# Embedding
# =========================================================
def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def estimate_text_lengths(tokenizer, texts: List[str], max_length: int) -> List[int]:
    lengths = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append(len(encoded["input_ids"]))
    return lengths


def build_length_grouped_order(
    lengths: List[int],
    batch_size: int,
    shuffle_batches: bool = False,
    seed: int = 42,
) -> List[int]:
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = [
        sorted_indices[i:i + batch_size]
        for i in range(0, len(sorted_indices), batch_size)
    ]
    if shuffle_batches:
        rng = random.Random(seed)
        rng.shuffle(batches)
    return [i for batch in batches for i in batch]


def load_embedding_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device


@torch.inference_mode()
def encode_texts(
    tokenizer,
    model,
    device,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    normalize: bool = True,
    length_sort: bool = True,
    shuffle_length_grouped_batches: bool = False,
    length_sort_seed: int = 42,
) -> np.ndarray:
    """Compute mean-pooled embeddings, returned in the original input order."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if length_sort:
        lengths = estimate_text_lengths(tokenizer, texts, max_length)
        ordered_indices = build_length_grouped_order(
            lengths=lengths,
            batch_size=batch_size,
            shuffle_batches=shuffle_length_grouped_batches,
            seed=length_sort_seed,
        )
    else:
        ordered_indices = list(range(len(texts)))

    ordered_texts = [texts[i] for i in ordered_indices]

    all_embeds = []
    for start in range(0, len(ordered_texts), batch_size):
        batch_texts = ordered_texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        embeds = mean_pool_last_hidden(outputs.last_hidden_state, encoded["attention_mask"])
        if normalize:
            embeds = F.normalize(embeds, p=2, dim=-1)

        all_embeds.append(embeds.detach().float().cpu())

        del encoded, outputs, embeds
        cleanup_torch_memory()

    result = torch.cat(all_embeds, dim=0).numpy().astype(np.float32)

    restored = np.empty_like(result)
    for sorted_pos, original_idx in enumerate(ordered_indices):
        restored[original_idx] = result[sorted_pos]

    del all_embeds, result
    cleanup_torch_memory()
    return restored


# =========================================================
# Embedding cache (shared with eval_tofu.py)
# =========================================================
def load_questions_jsonl(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line)["text"])
    return questions


def make_full_cache_key(
    dataset_name: str,
    config: str,
    split: str,
    embedding_model_id: str,
    embed_max_length: int,
    normalize: bool,
) -> str:
    return "||".join([
        dataset_name,
        config,
        split,
        embedding_model_id,
        str(embed_max_length),
        str(int(normalize)),
    ])


def get_cache_paths(cache_root: str, cache_key: str) -> Dict[str, str]:
    cache_dir = os.path.join(cache_root, safe_filename(cache_key))
    return {
        "dir": cache_dir,
        "metadata": os.path.join(cache_dir, "metadata.json"),
        "questions_jsonl": os.path.join(cache_dir, "questions.jsonl"),
        "embeds_npy": os.path.join(cache_dir, "embeds.npy"),
    }


def try_load_cached_embeddings(
    *,
    cache_root: str,
    dataset_name: str,
    config: str,
    split: str,
    embedding_model_id: str,
    embed_max_length: int,
    normalize: bool,
    questions: List[str],
) -> Optional[np.ndarray]:
    """
    Return cached embeddings only if all cache files exist, the cached
    questions match the current questions exactly, and the row count matches.
    """
    cache_key = make_full_cache_key(
        dataset_name=dataset_name,
        config=config,
        split=split,
        embedding_model_id=embedding_model_id,
        embed_max_length=embed_max_length,
        normalize=normalize,
    )
    paths = get_cache_paths(cache_root, cache_key)

    if not (
        os.path.exists(paths["metadata"])
        and os.path.exists(paths["questions_jsonl"])
        and os.path.exists(paths["embeds_npy"])
    ):
        print(f"[CACHE MISS] Cache files not found: {paths['dir']}")
        return None

    try:
        if load_questions_jsonl(paths["questions_jsonl"]) != questions:
            print(f"[CACHE MISS] Question text mismatch: {paths['dir']}")
            return None

        X = np.load(paths["embeds_npy"])
        if X.shape[0] != len(questions):
            print(f"[CACHE MISS] Row count mismatch: {paths['dir']}")
            return None

        print(f"[CACHE HIT] Loaded embeddings from: {paths['embeds_npy']}")
        return X

    except Exception as e:
        print(f"[CACHE MISS] Failed to load cache from {paths['dir']}: {repr(e)}")
        return None


# =========================================================
# LLM for rule induction
# =========================================================
def load_causal_lm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    device = next(model.parameters()).device
    return tokenizer, model, device


def generate_chat(
    tokenizer,
    model,
    device,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 180,
) -> str:
    """Greedy decoding with the model's native chat template."""
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,  # Qwen3: disable thinking mode; ignored by other templates
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    terminators = [tokenizer.eos_token_id]
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama-3 end-of-turn
        if isinstance(eot, int) and eot != tokenizer.unk_token_id:
            terminators.append(eot)
    except Exception:
        pass

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    resp = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(resp, skip_special_tokens=True).strip()


# =========================================================
# Rule induction
# =========================================================
RULE_INDUCTION_SYSTEM_PROMPT = (
    "You are an expert in semantic clustering and rule-based routing for language models. "
    "Your task is to derive precise and discriminative rules from clustered user requests.\n\n"
    "You will be given a set of examples that all belong to the SAME cluster. "
    "Your goal is to summarize a rule that helps a model RECOGNIZE requests from this cluster in the future.\n"
    "The rule must achieve HIGH PRECISION:\n"
    "- It should correctly recognize requests that belong to this cluster.\n"
    "- It must NOT match requests from other clusters.\n"
    "Output ONLY the rule text. Do not include explanations, headings, or lists."
)


def clean_rule_text(text: str) -> str:
    """Strip common preambles, bullets and code fences from the generated rule."""
    t = (text or "").strip()
    t = re.sub(r"(?i)^system prompt instruction:\s*", "", t)
    t = re.sub(r"^\s*[-*•]+\s*", "", t)
    t = re.sub(r"^```.*?\n", "", t, flags=re.DOTALL)
    t = re.sub(r"\n```$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def induce_rule(
    tokenizer,
    model,
    device,
    cluster_examples: List[str],
    max_new_tokens: int = 180,
) -> str:
    examples_text = "\n".join([f"- {ex}" for ex in cluster_examples])

    messages = [
        {"role": "system", "content": RULE_INDUCTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""
I am providing a set of user request examples from the same cluster.

User request examples:
{examples_text}

Output ONLY the Rule text itself. Do not output anything else.
""".strip(),
        },
    ]

    raw = generate_chat(tokenizer, model, device, messages, max_new_tokens=max_new_tokens)
    return clean_rule_text(raw)


def top_representative_examples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    questions: List[str],
    cluster_id: int,
    topn: int = 20,
) -> List[str]:
    """Return up to `topn` questions of a cluster, ranked by cosine similarity to its centroid."""
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) == 0:
        return []

    cluster_embeds = embeddings[cluster_indices]
    centroid = cluster_embeds.mean(axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12)

    sims = (cluster_embeds @ centroid.T).squeeze(-1)
    order = np.argsort(-sims)[:topn]
    return [questions[cluster_indices[i]] for i in order]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cluster TOFU forget questions and induce one refusal rule per cluster."
    )

    parser.add_argument(
        "--embedding_model_id",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model used for clustering and cache lookup.",
    )
    parser.add_argument(
        "--induction_model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="LLM that induces the rules (the same LLM that applies them at inference).",
    )

    parser.add_argument("--dataset_name", type=str, default="locuslab/TOFU")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="forget05",
        choices=["forget01", "forget05", "forget10"],
        help="TOFU forget split.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters (= number of rules).")

    parser.add_argument(
        "--sample_per_cluster",
        type=int,
        default=10,
        help="Number of examples shown to the LLM per cluster.",
    )
    parser.add_argument(
        "--rep_pool_size",
        type=int,
        default=20,
        help="Examples are sampled from the max(sample_per_cluster, rep_pool_size) "
             "questions closest to the cluster centroid.",
    )
    parser.add_argument(
        "--question_char_limit",
        type=int,
        default=512,
        help="Truncate each example question to this many characters in the prompt.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap on the number of questions (for debugging).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default=None,
        help="Output file. Default: rules_tofu_{dataset_config}_k{k}.jsonl",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it.",
    )

    parser.add_argument("--embed_batch_size", type=int, default=2)
    parser.add_argument("--embed_max_length", type=int, default=512)
    parser.add_argument(
        "--cache_root",
        type=str,
        default="./embedding_cache_tofu",
        help="Embedding cache directory shared with eval_tofu.py.",
    )

    # Only affect on-the-fly embedding; not part of the cache key.
    parser.add_argument("--no_length_sort", action="store_true",
                        help="Disable length-grouped batching when computing embeddings without the cache.")
    parser.add_argument("--shuffle_length_grouped_batches", action="store_true")
    parser.add_argument("--length_sort_seed", type=int, default=42)

    args = parser.parse_args()

    normalize = True
    length_sort = not args.no_length_sort
    out_jsonl = args.out_jsonl or f"rules_tofu_{args.dataset_config}_k{args.k}.jsonl"

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.append and os.path.exists(out_jsonl):
        os.remove(out_jsonl)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    write_jsonl(
        out_jsonl,
        {
            "_meta": True,
            "created_at_unix": int(time.time()),
            "embedding_model_id": args.embedding_model_id,
            "induction_model_id": args.induction_model_id,
            "dataset": {
                "name": args.dataset_name,
                "config": args.dataset_config,
                "split": args.split,
            },
            "k": args.k,
            "sample_per_cluster": args.sample_per_cluster,
            "rep_pool_size": args.rep_pool_size,
            "seed": args.seed,
            "embedding_batch_size": args.embed_batch_size,
            "embedding_max_length": args.embed_max_length,
        },
    )

    # ---------------- Data ----------------
    ds = load_dataset_items(args.dataset_name, args.dataset_config, args.split)
    questions, original_indices = build_questions(ds, args.max_questions)

    print(f">>> Loaded {len(questions)} questions")
    if len(questions) < args.k:
        raise ValueError(f"Not enough questions ({len(questions)}) for k={args.k}")

    # ---------------- Embeddings ----------------
    X = try_load_cached_embeddings(
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        config=args.dataset_config,
        split=args.split,
        embedding_model_id=args.embedding_model_id,
        embed_max_length=args.embed_max_length,
        normalize=normalize,
        questions=questions,
    )

    if X is None:
        print(f">>> No usable cache. Computing embeddings with: {args.embedding_model_id}")
        emb_tokenizer = emb_model = None
        try:
            emb_tokenizer, emb_model, emb_device = load_embedding_model(args.embedding_model_id)
            X = encode_texts(
                emb_tokenizer,
                emb_model,
                emb_device,
                questions,
                batch_size=args.embed_batch_size,
                max_length=args.embed_max_length,
                normalize=normalize,
                length_sort=length_sort,
                shuffle_length_grouped_batches=args.shuffle_length_grouped_batches,
                length_sort_seed=args.length_sort_seed,
            )
        finally:
            del emb_model, emb_tokenizer
            cleanup_torch_memory()
    else:
        print(">>> Using cached embeddings; skipping embedding model loading.")

    print(f">>> Embedding matrix shape: {X.shape}")

    # ---------------- Clustering ----------------
    print(f">>> KMeans clustering (k={args.k}) ...")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(X)

    cluster_to_items: Dict[int, List[Tuple[int, str]]] = {c: [] for c in range(args.k)}
    for idx_in_list, c in enumerate(labels):
        cluster_to_items[int(c)].append((original_indices[idx_in_list], questions[idx_in_list]))

    # ---------------- Rule induction ----------------
    print(f">>> Loading rule induction model: {args.induction_model_id}")
    tokenizer = model = None
    try:
        tokenizer, model, device = load_causal_lm(args.induction_model_id)

        for c in range(args.k):
            items = cluster_to_items[c]
            if not items:
                continue

            cluster_indices = np.where(labels == c)[0]
            centroid = X[cluster_indices].mean(axis=0)

            reps = top_representative_examples(
                X,
                labels,
                questions,
                c,
                topn=max(args.sample_per_cluster, args.rep_pool_size),
            )
            rng = random.Random(args.seed + c)
            if len(reps) <= args.sample_per_cluster:
                sampled_questions = reps
            else:
                sampled_questions = rng.sample(reps, args.sample_per_cluster)

            examples = [truncate(q, args.question_char_limit) for q in sampled_questions]
            rule_text = induce_rule(
                tokenizer,
                model,
                device,
                examples,
                max_new_tokens=args.max_new_tokens,
            )

            write_jsonl(
                out_jsonl,
                {
                    "rule_id": f"k{args.k}_c{c:03d}",
                    "cluster_id": c,
                    "cluster_size": len(items),
                    "dataset_index_min": min(i for i, _ in items),
                    "dataset_index_max": max(i for i, _ in items),
                    "cluster_embedding_dim": int(centroid.shape[0]),
                    "cluster_embedding_center": centroid.tolist(),
                    "rule_text": rule_text,
                    "created_at_unix": int(time.time()),
                    "embedding_model_id": args.embedding_model_id,
                    "induction_model_id": args.induction_model_id,
                },
            )
            print(f"[cluster {c:03d}] size={len(items)}  rule: {truncate(rule_text, 120)}")

    finally:
        del model, tokenizer
        cleanup_torch_memory()

    print(f"\nDone. Rules saved to: {out_jsonl}")
    print(f"Mode: {'append' if args.append else 'overwrite'}")
    print(f"Embedding model: {args.embedding_model_id}")
    print(f"Induction model: {args.induction_model_id}")


if __name__ == "__main__":
    main()
