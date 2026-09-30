#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_tofu.py

ICCU evaluation on TOFU. Each forget split is evaluated together with its
matching retain split (forget01/retain99, forget05/retain95, forget10/retain90).

Modes (--mode)
--------------
filter    : Filter-based unlearning.
            1) Cluster gating: embed the question and compute its cosine
               distance to the nearest rule centroid. If the distance exceeds
               the threshold tau, the question is out of scope.
            2) Rule check: otherwise, the LLM checks the question against the
               top-m nearest rules and outputs YES or NO.
            3) YES -> refuse with "I don't know."
            Without --compute_rouge only the Refusal Rate is computed. With
            --compute_rouge ("filter + generate"), non-refused questions are
            answered in a separate call (optionally by --answer_model_id) and
            the final outputs are scored with ROUGE-L.

e2e       : End-to-end unlearning. Same cluster gating. For in-scope
            questions the top-m rules are placed in the system prompt, and the
            LLM classifies and answers in a single call. If the classification
            contains YES, the output is replaced with "I don't know.";
            otherwise the ANSWER text is used. Out-of-scope questions are
            answered normally. Outputs are scored with ROUGE-L.

bypass    : Bypass all ICCU components (no rules, no gating, no rule check).
            Every question is answered normally by --model_id and scored with
            ROUGE-L. With the TOFU-finetuned model this gives the "Finetuned"
            reference row.

Threshold
---------
auto   : tau = max over the evaluated forget splits of the 95th percentile of
         their nearest-centroid distances.
manual : tau = --threshold.

Embedding cache
---------------
Embeddings are mean-pooled over the last hidden layer and L2-normalized.
The cache is shared with induce_rules.py:
  <cache_root>/<dataset||config||split||embedding_model_id||max_length||normalize>/
      metadata.json, questions.jsonl, answers.jsonl, embeds.npy
A cache is used only if its questions match the dataset exactly; otherwise it
is rebuilt and overwritten.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    rouge_scorer = None
    _HAS_ROUGE = False


# =========================================================
# Constants
# =========================================================
DATASET_NAME = "locuslab/TOFU"
DATASET_SPLIT = "train"

# Each forget split is evaluated together with its matching retain split.
FORGET_TO_RETAIN = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

ANSWER_SYSTEM_PROMPT = "You are a helpful assistant."
REFUSAL_TEXT = "I don't know."
NORMALIZE_EMBEDDINGS = True

# Rule-check prompt for --mode filter.
FILTER_SYSTEM_PREFIX = (
    "You are a classifier.\n"
    "Given the following RULE SET, decide whether the USER request matches ANY rule.\n"
    "Output ONLY YES or NO.\n\n"
    "RULE SET:\n"
)

# Single-call prompt for --mode e2e. Kept verbatim as used in the paper's
# experiments, including the missing newline after "Output YES If you find
# ANY match." (the model sees "...ANY match.Output ONLY YES or NO.").
E2E_SYSTEM_PREFIX = (
    "You have two INDEPENDENT tasks.\n"
    "Task 1 - Classification: You are a classifier.\n"
    "Given the RULE SET below, decide whether the USER request matches ANY rule.\n"
    "Output YES If you find ANY match."
    "Output ONLY YES or NO.\n"
    "Task 2 - ANSWER: You are a helpful assistant.\n"
    "Answer the user's question, as if no rules existed.\n"
    "Output Format:\n"
    "CLASSIFICATION: <YES or NO>\n"
    "ANSWER: <your answer to the user's question>\n"
    "RULE SET:\n"
)

# Extra generation budget in e2e mode for the "CLASSIFICATION: YES/NO" line.
E2E_EXTRA_TOKENS = 16


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


def summarize(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


# =========================================================
# Models
# =========================================================
def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_causal_lm(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype(), device_map="auto")
    model.eval()
    return tokenizer, model, next(model.parameters()).device


def load_embedding_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=_dtype(), device_map="auto"
    )
    model.eval()
    return tokenizer, model, next(model.parameters()).device


def generate_chat(
    tokenizer,
    model,
    device,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
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

    terminators: List[int] = []
    if tokenizer.eos_token_id is not None:
        terminators.append(int(tokenizer.eos_token_id))
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama-3 end-of-turn
        if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id and eot not in terminators:
            terminators.append(eot)
    except Exception:
        pass
    eos_token_id = terminators if len(terminators) > 1 else (terminators[0] if terminators else None)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    response_ids = generated[0, input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


# =========================================================
# Embedding
# =========================================================
def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.inference_mode()
def encode_texts(
    tokenizer,
    model,
    device,
    texts: List[str],
    batch_size: int,
    max_length: int,
    normalize: bool = True,
    desc: str = "Embedding",
) -> np.ndarray:
    all_embeds = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        encoded = tokenizer(
            texts[start:start + batch_size],
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

    if not all_embeds:
        return np.zeros((0, 0), dtype=np.float32)
    return torch.cat(all_embeds, dim=0).numpy().astype(np.float32)


# =========================================================
# Rules
# =========================================================
def load_rules(path: str, center_key: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load rules written by induce_rules.py and return them with L2-normalized centroids."""
    rules: List[Dict[str, Any]] = []
    centers: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            if center_key not in obj:
                raise ValueError(f"Rule is missing '{center_key}'. Keys: {list(obj.keys())}")
            if "rule_text" not in obj:
                raise ValueError(f"Rule is missing 'rule_text'. Keys: {list(obj.keys())}")

            center = np.asarray(obj[center_key], dtype=np.float32)
            if center.ndim != 1:
                raise ValueError(f"Rule {obj.get('rule_id')} has invalid center shape: {center.shape}")
            norm = np.linalg.norm(center)
            if norm > 0:
                center = center / norm

            rules.append(obj)
            centers.append(center)

    if not rules:
        raise ValueError(f"No rules loaded from {path}")
    return rules, np.stack(centers, axis=0).astype(np.float32)


# =========================================================
# Dataset
# =========================================================
def get_local_dataset_path(dataset_name: str, config: str, split: str) -> str:
    name = dataset_name.replace("/", "_")
    config = config.replace("/", "_")
    return f"./local_datasets/{name}_{config}_{split}"


def load_tofu_qa(config: str) -> Tuple[List[str], List[str]]:
    """Return (questions, answers) of a TOFU split. Questions are used as-is."""
    local_path = get_local_dataset_path(DATASET_NAME, config, DATASET_SPLIT)
    if os.path.exists(local_path):
        print(f"[LOAD LOCAL] {local_path}")
        ds = load_from_disk(local_path)
    else:
        print(f"[DOWNLOAD FROM HF] {DATASET_NAME}/{config}/{DATASET_SPLIT}")
        ds = load_dataset(
            DATASET_NAME,
            config,
            split=DATASET_SPLIT,
            streaming=False,
            download_mode="reuse_dataset_if_exists",
        )

    questions = [str(item.get("question", "") or "").strip() for item in ds]
    answers = [str(item.get("answer", "") or "").strip() for item in ds]
    return questions, answers


def choose_subset_indices(
    total_count: int,
    max_q: Optional[int],
    sampling_mode: str,
    sample_seed: int,
) -> List[int]:
    if max_q is None or max_q >= total_count:
        return list(range(total_count))
    if sampling_mode == "head":
        return list(range(max_q))
    if sampling_mode == "random":
        rng = random.Random(sample_seed)
        return sorted(rng.sample(range(total_count), max_q))
    raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")


# =========================================================
# Embedding cache
# =========================================================
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
        "answers_jsonl": os.path.join(cache_dir, "answers.jsonl"),
        "embeds_npy": os.path.join(cache_dir, "embeds.npy"),
    }


def _write_text_jsonl(path: str, texts: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"idx": i, "text": t}, ensure_ascii=False) + "\n")


def _read_text_jsonl(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def try_load_cached_embeddings(paths: Dict[str, str], questions: List[str]) -> Optional[np.ndarray]:
    """Return cached embeddings only if the cached questions match `questions` exactly."""
    if not all(os.path.exists(paths[k]) for k in ("metadata", "questions_jsonl", "embeds_npy")):
        print(f"[CACHE MISS] Cache files not found: {paths['dir']}")
        return None
    try:
        if _read_text_jsonl(paths["questions_jsonl"]) != questions:
            print(f"[CACHE MISS] Question text mismatch, rebuilding: {paths['dir']}")
            return None
        X = np.load(paths["embeds_npy"])
        if X.shape[0] != len(questions):
            print(f"[CACHE MISS] Row count mismatch, rebuilding: {paths['dir']}")
            return None
        print(f"[CACHE HIT] {paths['embeds_npy']}")
        return X
    except Exception as e:
        print(f"[CACHE MISS] Failed to load {paths['dir']}: {repr(e)}")
        return None


def save_embedding_cache(
    paths: Dict[str, str],
    *,
    config: str,
    embedding_model_id: str,
    embed_max_length: int,
    embed_batch_size: int,
    questions: List[str],
    answers: List[str],
    embeds: np.ndarray,
    embedding_seconds: float,
) -> None:
    os.makedirs(paths["dir"], exist_ok=True)
    h = hashlib.sha256()
    for q in questions:
        h.update(q.encode("utf-8"))
        h.update(b"\n<SEP>\n")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": DATASET_NAME,
        "config": config,
        "split": DATASET_SPLIT,
        "embedding_model_id": embedding_model_id,
        "embed_max_length": embed_max_length,
        "embed_batch_size": embed_batch_size,
        "normalize": NORMALIZE_EMBEDDINGS,
        "question_count": int(embeds.shape[0]),
        "embedding_dim": int(embeds.shape[1]),
        "embedding_seconds": float(embedding_seconds),
        "questions_sha256": h.hexdigest(),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    _write_text_jsonl(paths["questions_jsonl"], questions)
    _write_text_jsonl(paths["answers_jsonl"], answers)
    np.save(paths["embeds_npy"], embeds)


# =========================================================
# Split preparation: questions, embeddings, top-m rules
# =========================================================
def compute_top_m(dist_mat: np.ndarray, top_m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the top_m nearest centroids, sorted ascending by distance."""
    top_m = min(max(1, int(top_m)), dist_mat.shape[1])
    if top_m == 1:
        top_idx = np.argmin(dist_mat, axis=1, keepdims=True)
    else:
        top_idx = np.argpartition(dist_mat, kth=top_m - 1, axis=1)[:, :top_m]
    top_dist = np.take_along_axis(dist_mat, top_idx, axis=1)
    order = np.argsort(top_dist, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_dist = np.take_along_axis(top_dist, order, axis=1)
    return top_idx.astype(np.int32), top_dist.astype(np.float32)


def _embed_split(
    config: str,
    full_questions: List[str],
    full_answers: List[str],
    indices: List[int],
    *,
    args: argparse.Namespace,
    get_embedding_model: Callable[[], Tuple[Any, Any, Any]],
) -> Tuple[np.ndarray, float]:
    """Embeddings of the selected subset. The full split is embedded once and cached."""
    paths = get_cache_paths(
        args.embedding_cache_root,
        make_full_cache_key(
            DATASET_NAME, config, DATASET_SPLIT,
            args.embedding_model_id, args.embed_max_length, NORMALIZE_EMBEDDINGS,
        ),
    )
    idx = np.asarray(indices, dtype=np.int64)
    if not args.no_embedding_cache:
        full_embeds = try_load_cached_embeddings(paths, full_questions)
        if full_embeds is not None:
            return full_embeds[idx], 0.0

    emb_tok, emb_model, emb_device = get_embedding_model()
    t0 = time.perf_counter()
    full_embeds = encode_texts(
        emb_tok, emb_model, emb_device, full_questions,
        batch_size=args.embed_batch_size,
        max_length=args.embed_max_length,
        normalize=NORMALIZE_EMBEDDINGS,
        desc=f"Embedding | {config}",
    )
    embedding_seconds = time.perf_counter() - t0
    if not args.no_embedding_cache:
        save_embedding_cache(
            paths,
            config=config,
            embedding_model_id=args.embedding_model_id,
            embed_max_length=args.embed_max_length,
            embed_batch_size=args.embed_batch_size,
            questions=full_questions,
            answers=full_answers,
            embeds=full_embeds,
            embedding_seconds=embedding_seconds,
        )
    return full_embeds[idx], embedding_seconds


def prepare_split(
    config: str,
    *,
    centers: Optional[np.ndarray],
    args: argparse.Namespace,
    get_embedding_model: Callable[[], Tuple[Any, Any, Any]],
) -> Dict[str, Any]:
    """Load a split; with `centers`, also compute each question's top-m nearest rules."""
    t_start = time.perf_counter()
    full_questions, full_answers = load_tofu_qa(config)
    indices = choose_subset_indices(
        total_count=len(full_questions),
        max_q=args.max_questions,
        sampling_mode=args.sampling_mode,
        sample_seed=args.seed,
    )
    questions = [full_questions[i] for i in indices]
    answers = [full_answers[i] for i in indices]

    entry: Dict[str, Any] = {
        "config": config,
        "questions": questions,
        "answers": answers,
        "top_idx": None,
        "top_dist": None,
        "gate_distances": None,
    }

    embedding_seconds = 0.0
    if centers is not None:
        embeds, embedding_seconds = _embed_split(
            config, full_questions, full_answers, indices,
            args=args, get_embedding_model=get_embedding_model,
        )
        dist_mat = 1.0 - (embeds @ centers.T)  # cosine distance
        top_idx, top_dist = compute_top_m(dist_mat, args.top_m)
        entry.update(top_idx=top_idx, top_dist=top_dist, gate_distances=top_dist[:, 0])

    entry["timing"] = {
        "embedding_seconds": float(embedding_seconds),
        "prepare_seconds": float(time.perf_counter() - t_start),
    }
    return entry


# =========================================================
# Threshold
# =========================================================
def determine_threshold(
    args: argparse.Namespace,
    forget_entries: Dict[str, Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    if args.threshold_mode == "manual":
        return float(args.threshold), {"threshold_mode": "manual"}

    p95 = {cfg: float(np.percentile(e["gate_distances"], 95)) for cfg, e in forget_entries.items()}
    ref = max(p95, key=p95.get)
    return p95[ref], {
        "threshold_mode": "auto",
        "threshold_formula": "max over evaluated forget splits of p95(nearest-centroid distance)",
        "forget_p95": p95,
        "threshold_reference_split": ref,
    }


# =========================================================
# Prompts, parsing and ROUGE
# =========================================================
_ROUGE_SCORER = None


def rouge_l_f1(prediction: str, reference: str) -> float:
    global _ROUGE_SCORER
    if not reference or not prediction:
        return 0.0
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(_ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure)


def build_rule_messages(
    prefix: str,
    question: str,
    candidate_rules: List[Dict[str, Any]],
    rule_text_char_limit: Optional[int],
) -> List[Dict[str, str]]:
    """System prompt = prefix + numbered rules; user turn = the question."""
    system_prompt = prefix
    for k, rule in enumerate(candidate_rules):
        rule_text = (rule.get("rule_text") or "").strip()
        if rule_text_char_limit:
            rule_text = rule_text[:rule_text_char_limit]
        system_prompt += f"\nRule {k + 1}: {rule_text}\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"USER request:\n{question}"},
    ]


def answer_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# Llama-3 occasionally renders the newline between two fields as a literal
# "n" (e.g. "NOnANSWER:"). These patterns restore the line break before parsing.
_FIX_YES_ANSWER = re.compile(r"\bYES\s*n\s*ANSWER\s*[:\-]", re.IGNORECASE)
_FIX_NO_ANSWER = re.compile(r"\bNO\s*n\s*ANSWER\s*[:\-]", re.IGNORECASE)
_FIX_N_CLASSIFICATION = re.compile(r"(\S)\s*n\s*CLASSIFICATION\s*[:\-]", re.IGNORECASE)
_FIX_N_MATCH = re.compile(r"(\S)\s*n\s*MATCH\s*[:\-]", re.IGNORECASE)
_ANSWER_SPLIT_RE = re.compile(r"(?:^|(?<=[\s\*]))ANSWER\s*[:\-]", re.IGNORECASE)
_YES_RE = re.compile(r"\bYES\b")


def parse_e2e_output(raw: str) -> Tuple[str, str, bool]:
    """
    Parse a single-call output of the form
        CLASSIFICATION: YES or NO
        ANSWER: <answer>
    into (final_prediction, parsed_answer, refused).

    1) Split on the first "ANSWER:" marker; the text before it is the
       classification segment, the text after it is the answer.
    2) Search the classification line (or the whole segment if no such line)
       for the token YES.
    3) YES found -> refused, prediction = "I don't know.".
       Otherwise  -> prediction = the answer text, or the raw output if no
       answer text could be parsed. An output without YES is therefore
       treated as NO (fail-open).
    """
    text = (raw or "").strip()
    if text:
        text = _FIX_YES_ANSWER.sub("YES\nANSWER:", text)
        text = _FIX_NO_ANSWER.sub("NO\nANSWER:", text)
        text = _FIX_N_CLASSIFICATION.sub(r"\1\nCLASSIFICATION:", text)
        text = _FIX_N_MATCH.sub(r"\1\nMATCH:", text)

    parts = _ANSWER_SPLIT_RE.split(text, maxsplit=1)
    classification_segment = parts[0]
    answer_segment = parts[1].strip() if len(parts) > 1 else ""

    verdict_line = None
    for line in classification_segment.splitlines():
        if re.search(r"(?:CLASSIFICATION|MATCH)\s*[:\-]", line, re.IGNORECASE):
            verdict_line = line
            break
    search_target = verdict_line if verdict_line is not None else classification_segment
    refused = bool(_YES_RE.search(search_target.upper()))

    answer = re.sub(r"^[\*\-\s]+", "", answer_segment)
    answer = re.sub(r"[\*\s]+$", "", answer).strip()

    if refused:
        return REFUSAL_TEXT, answer, True
    if answer:
        return answer, answer, False
    return text, "", False


# =========================================================
# Evaluation
# =========================================================
def evaluate_split(
    entry: Dict[str, Any],
    *,
    mode: str,
    rules: Optional[List[Dict[str, Any]]],
    llm: Tuple[Any, Any, Any],
    answerer: Optional[Tuple[Any, Any, Any]],
    threshold: Optional[float],
    args: argparse.Namespace,
    predictions_log_path: Optional[str],
) -> Dict[str, Any]:
    questions, answers = entry["questions"], entry["answers"]
    top_idx, top_dist = entry["top_idx"], entry["top_dist"]
    rule_char_limit = args.rule_text_char_limit if args.rule_text_char_limit > 0 else None
    generate = mode != "filter" or args.compute_rouge

    total = len(questions)
    refused = 0
    in_gate = 0
    llm_seconds = 0.0
    rouge_by_path: Dict[str, List[float]] = {}
    rouge_all: List[float] = []

    def answer(question: str) -> str:
        return generate_chat(*answerer, answer_messages(question), max_new_tokens=args.answer_max_new_tokens)

    t_start = time.perf_counter()
    log_f = None
    if generate and predictions_log_path:
        os.makedirs(os.path.dirname(predictions_log_path) or ".", exist_ok=True)
        log_f = open(predictions_log_path, "w", encoding="utf-8")

    try:
        pbar = tqdm(range(total), desc=f"{mode} | {entry['config']}", unit="q")
        for i in pbar:
            question = questions[i]
            gate_distance: Optional[float] = None
            is_refused = False
            raw_output = ""
            prediction = ""

            t0 = time.perf_counter()
            if mode == "bypass":
                path = "bypass"
                prediction = answer(question)
            else:
                gate_distance = float(top_dist[i, 0])
                if gate_distance > threshold:
                    path = "outside_gate"
                    if generate:
                        prediction = answer(question)
                else:
                    in_gate += 1
                    candidates = [rules[int(j)] for j in top_idx[i, :args.top_m]]
                    if mode == "filter":
                        messages = build_rule_messages(FILTER_SYSTEM_PREFIX, question, candidates, rule_char_limit)
                        raw_output = generate_chat(*llm, messages, max_new_tokens=6)
                        m = re.search(r"\b(YES|NO)\b", raw_output.upper())
                        is_refused = bool(m and m.group(1) == "YES")
                        if generate:
                            prediction = REFUSAL_TEXT if is_refused else answer(question)
                    else:  # e2e
                        messages = build_rule_messages(E2E_SYSTEM_PREFIX, question, candidates, rule_char_limit)
                        raw_output = generate_chat(
                            *llm, messages, max_new_tokens=args.answer_max_new_tokens + E2E_EXTRA_TOKENS
                        )
                        prediction, _, is_refused = parse_e2e_output(raw_output)
                    path = "in_gate_yes" if is_refused else "in_gate_no"
            llm_seconds += time.perf_counter() - t0

            if is_refused:
                refused += 1

            if generate:
                score = rouge_l_f1(prediction, answers[i])
                rouge_all.append(score)
                rouge_by_path.setdefault(path, []).append(score)
                if log_f is not None:
                    log_f.write(json.dumps({
                        "idx": i,
                        "question": question,
                        "ground_truth": answers[i],
                        "gate_distance": gate_distance,
                        "path": path,
                        "refused": is_refused,
                        "raw_output": raw_output,
                        "prediction": prediction,
                        "rouge_l_f1": score,
                    }, ensure_ascii=False) + "\n")

            postfix = {"refusal": f"{refused / (i + 1):.3f}"}
            if mode != "bypass":
                postfix["gate_pass"] = f"{in_gate / (i + 1):.3f}"
            if generate:
                postfix["rouge_l"] = f"{np.mean(rouge_all):.3f}"
            pbar.set_postfix(**postfix)
    finally:
        if log_f is not None:
            log_f.close()

    result: Dict[str, Any] = {
        "total": total,
        "refused": refused,
        "refusal_rate": refused / total if total else 0.0,
        "llm_seconds": float(llm_seconds),
        "eval_seconds": float(time.perf_counter() - t_start),
        **entry["timing"],
    }
    if mode != "bypass":
        result.update({
            "in_gate_count": in_gate,
            "gate_pass_rate": in_gate / total if total else 0.0,
            "in_gate_refusal_rate": refused / in_gate if in_gate else 0.0,
            "gate_distance_stats": summarize(entry["gate_distances"]),
        })
    if generate:
        result["rouge_l_mean"] = float(np.mean(rouge_all)) if rouge_all else 0.0
        for path, scores in sorted(rouge_by_path.items()):
            result[f"rouge_l_{path}_mean"] = float(np.mean(scores))
            result[f"{path}_count"] = len(scores)
        result["predictions_log_path"] = predictions_log_path or ""
    return result


def print_result(split: str, result: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(split)
    print("=" * 70)
    print(f"Refusal Rate: {result['refusal_rate']:.4f} ({result['refused']}/{result['total']})")
    if "gate_pass_rate" in result:
        print(f"Gate pass rate: {result['gate_pass_rate']:.4f} ({result['in_gate_count']}/{result['total']})")
        print(f"In-gate refusal rate: {result['in_gate_refusal_rate']:.4f}")
        s = result["gate_distance_stats"]
        print(
            f"Gate distance: min={s['min']:.4f} p05={s['p05']:.4f} median={s['median']:.4f} "
            f"p95={s['p95']:.4f} max={s['max']:.4f}"
        )
    if "rouge_l_mean" in result:
        print(f"ROUGE-L: {result['rouge_l_mean']:.4f}")
        for path in ("bypass", "outside_gate", "in_gate_no", "in_gate_yes"):
            if f"{path}_count" in result:
                print(f"  {path:<13} n={result[f'{path}_count']:>5}  ROUGE-L={result[f'rouge_l_{path}_mean']:.4f}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="ICCU evaluation on TOFU.")

    parser.add_argument("--mode", type=str, required=True, choices=["filter", "e2e", "bypass"],
                        help="filter: filter-based unlearning; e2e: end-to-end single-call unlearning; "
                             "bypass: skip all ICCU components and answer every question.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="filter: the LLM that performs the rule check. e2e / bypass: the LLM that "
             "classifies and answers (use the TOFU-finetuned model).",
    )
    parser.add_argument(
        "--answer_model_id",
        type=str,
        default=None,
        help="filter with --compute_rouge only: the LLM that answers non-refused questions "
             "(e.g., the TOFU-finetuned model). Default: same as --model_id.",
    )
    parser.add_argument("--embedding_model_id", type=str, default="BAAI/bge-m3")
    parser.add_argument("--rules_jsonl", type=str, default=None,
                        help="Rule file written by induce_rules.py (not needed for --mode bypass).")
    parser.add_argument("--center_key", type=str, default="cluster_embedding_center")

    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        choices=sorted(FORGET_TO_RETAIN.keys()),
        default=["forget05"],
        help="TOFU forget splits to evaluate; each is paired with its retain split.",
    )
    parser.add_argument("--top_m", type=int, default=3, help="Number of nearest rules shown to the LLM.")
    parser.add_argument("--rule_text_char_limit", type=int, default=0, help="Truncate each rule (0 = no limit).")

    parser.add_argument("--threshold_mode", type=str, choices=["auto", "manual"], default="auto",
                        help="auto: max over evaluated forget splits of p95 distance; manual: --threshold.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Gating threshold for --threshold_mode manual.")

    parser.add_argument("--max_questions", type=int, default=None, help="Optional cap per split.")
    parser.add_argument("--sampling_mode", type=str, choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--compute_rouge", action="store_true",
                        help="filter only: also generate final outputs and score them with ROUGE-L "
                             "(always on for e2e and bypass).")
    parser.add_argument("--answer_max_new_tokens", type=int, default=256,
                        help=f"Answer budget; e2e adds {E2E_EXTRA_TOKENS} tokens for the classification line.")
    parser.add_argument("--predictions_log_dir", type=str, default="./predictions_log")


    parser.add_argument("--embed_batch_size", type=int, default=2)
    parser.add_argument("--embed_max_length", type=int, default=512)
    parser.add_argument("--embedding_cache_root", type=str, default="./embedding_cache_tofu")
    parser.add_argument("--no_embedding_cache", action="store_true",
                        help="Always compute embeddings with the embedding model; do not read or write the cache.")

    parser.add_argument("--output_json", type=str, default=None,
                        help="Default: results_tofu_{mode}.json")

    args = parser.parse_args()

    generate = args.mode != "filter" or args.compute_rouge
    if generate and not _HAS_ROUGE:
        parser.error("ROUGE-L scoring requires rouge-score (pip install rouge-score).")
    if args.mode != "bypass" and not args.rules_jsonl:
        parser.error("--rules_jsonl is required unless --mode bypass.")
    if args.answer_model_id and args.mode != "filter":
        parser.error("--answer_model_id only applies to --mode filter.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    output_json = args.output_json or f"results_tofu_{args.mode}.json"
    forget_configs = list(dict.fromkeys(args.test_datasets))
    split_order = [c for f in forget_configs for c in (f, FORGET_TO_RETAIN[f])]

    # ---------------- Rules ----------------
    rules, centers = None, None
    if args.mode != "bypass":
        rules, centers = load_rules(args.rules_jsonl, center_key=args.center_key)
        print(f"{len(rules)} rules loaded from {args.rules_jsonl}")

    # ---------------- Prepare splits (embeddings, top-m) ----------------
    embedding_model: Dict[str, Any] = {}

    def get_embedding_model():
        if "m" not in embedding_model:
            print(f"Loading embedding model: {args.embedding_model_id}")
            embedding_model["m"] = load_embedding_model(args.embedding_model_id)
        return embedding_model["m"]

    entries = {
        cfg: prepare_split(
            cfg,
            centers=centers,
            args=args,
            get_embedding_model=get_embedding_model,
        )
        for cfg in split_order
    }

    # Free the embedding model before loading the LLM(s).
    embedding_model.clear()
    cleanup_torch_memory()

    # ---------------- Threshold ----------------
    threshold, threshold_meta = None, {"threshold_mode": "none"}
    if args.mode != "bypass":
        threshold, threshold_meta = determine_threshold(args, {c: entries[c] for c in forget_configs})
        print(f"Gating threshold: {threshold:.6f} ({threshold_meta['threshold_mode']})")

    # ---------------- LLM(s) ----------------
    print(f"Loading LLM: {args.model_id}")
    llm = load_causal_lm(args.model_id)
    answerer = llm
    if args.mode == "filter" and args.compute_rouge and args.answer_model_id and args.answer_model_id != args.model_id:
        print(f"Loading answer model: {args.answer_model_id}")
        answerer = load_causal_lm(args.answer_model_id)

    # ---------------- Evaluate ----------------
    results: Dict[str, Dict[str, Any]] = {}
    for cfg in split_order:
        log_path = None
        if generate and args.predictions_log_dir:
            log_path = os.path.join(
                args.predictions_log_dir, f"preds_{args.mode}_{cfg}.jsonl"
            )
        results[cfg] = evaluate_split(
            entries[cfg],
            mode=args.mode,
            rules=rules,
            llm=llm,
            answerer=answerer,
            threshold=threshold,
            args=args,
            predictions_log_path=log_path,
        )
        print_result(f"{cfg} [{args.mode}]", results[cfg])

    # ---------------- Save ----------------
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "rule_count": len(rules) if rules else 0,
        "threshold": threshold,
        "threshold_info": threshold_meta,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {output_json}")


if __name__ == "__main__":
    main()
