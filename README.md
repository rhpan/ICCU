# ICCU: In-Context Continual Unlearning via Pattern-Induced Refusal Rules

Code for the AACL-IJCNLP 2026 paper
**[ICCU: In-Context Continual Unlearning via Pattern-Induced Refusal Rules](https://arxiv.org/abs/2605.27138)**
by Ruihao Pan and Suhang Wang (The Pennsylvania State University).

ICCU handles a stream of unlearning requests without updating model weights.
Each request is turned into a small set of natural-language refusal rules, and
the rules of all requests are applied at inference time. New requests only add
rules, so earlier requests are not affected and a request can be revoked by
deleting its rules.

## How it works

**Offline, once per unlearning request** (`induce_rules.py`)

1. Embed the questions of the forget set and cluster them into K clusters.
2. For each cluster, show the LLM a few representative questions and ask it to
   summarize them into one refusal rule.
3. Store each rule together with its cluster centroid. The rule repository is
   the union of the rules of all requests so far.

**Online, for every query** (`eval_tofu.py`)

1. **Cluster gating.** Embed the query and compute its cosine distance to the
   nearest centroid. If the distance is above a threshold τ, the query is out
   of scope and is answered normally.
2. **Rule check.** Otherwise, the LLM checks the query against the top-m
   nearest rules. If it matches any rule, the model answers "I don't know.".

The rule check can be deployed in two modes:

| Mode | What happens for an in-scope query |
|---|---|
| `filter` | A separate YES/NO call decides whether to refuse; non-refused queries are answered in a second call. |
| `e2e` | The rules go into the system prompt and one call outputs both the decision and the answer. |

## Repository layout

```
.
├── induce_rules.py   # clustering + rule induction for one unlearning request
└── eval_tofu.py      # evaluation on TOFU (filter / e2e / bypass)
```

## Setup

```bash
conda create -n iccu python=3.10 -y
conda activate iccu
pip install torch transformers datasets numpy scikit-learn tqdm rouge-score
```

TOFU (`locuslab/TOFU`) and the models are downloaded from the Hugging Face Hub
on first use. Access to `meta-llama/Meta-Llama-3-8B-Instruct` requires
accepting its license on the Hub.

**TOFU-finetuned model.** The end-to-end and bypass modes, and answer
generation in filter mode, need a Llama-3-8B-Instruct model finetuned on the
full TOFU dataset.

```bash
FT_MODEL=/path/to/tofu-finetuned-llama3
```

## Usage

**1. Induce rules for each unlearning request**

```bash
python induce_rules.py --dataset_config forget01 --k 20 --out_jsonl rules/rules_forget01_k20.jsonl
python induce_rules.py --dataset_config forget05 --k 20 --out_jsonl rules/rules_forget05_k20.jsonl
python induce_rules.py --dataset_config forget10 --k 25 --out_jsonl rules/rules_forget10_k25.jsonl

# the rule repository is the union of all rule sets
cat rules/rules_forget01_k20.jsonl rules/rules_forget05_k20.jsonl \
    rules/rules_forget10_k25.jsonl > rules/rules_all.jsonl
```

Each output file starts with a `_meta` line recording the models and
hyperparameters, followed by one line per rule (`rule_text` and
`cluster_embedding_center`).

**2. Evaluate**

```bash
# filter: rule check with the off-the-shelf model, answers from the finetuned model
python eval_tofu.py --mode filter --compute_rouge --rules_jsonl rules/rules_all.jsonl \
    --answer_model_id $FT_MODEL --test_datasets forget01 forget05 forget10

# end-to-end
python eval_tofu.py --mode e2e --model_id $FT_MODEL --rules_jsonl rules/rules_all.jsonl \
    --test_datasets forget01 forget05 forget10

# bypass: no ICCU component, the finetuned model answers every question
python eval_tofu.py --mode bypass --model_id $FT_MODEL \
    --test_datasets forget01 forget05 forget10
```

Main options of `eval_tofu.py`:

| Option | Meaning |
|---|---|
| `--mode` | `filter`, `e2e`, or `bypass` |
| `--model_id` | `filter`: model for the YES/NO rule check. `e2e` / `bypass`: model that answers. |
| `--answer_model_id` | `filter` with `--compute_rouge` only: model that answers non-refused queries. Default: `--model_id`. |
| `--compute_rouge` | `filter` only: also generate answers and report ROUGE-L. Always on for `e2e` and `bypass`. |
| `--test_datasets` | Forget splits to evaluate. Each is paired with its retain split (forget01/retain99, forget05/retain95, forget10/retain90). |
| `--top_m` | Number of nearest rules shown to the LLM (default 3). |
| `--threshold_mode` | `auto` (default): τ is the largest 95th percentile of nearest-centroid distance over the evaluated forget splits. `manual`: use `--threshold`. |
| `--max_questions` | Optional per-split cap with random sampling (seed 42). |

Each run writes a JSON file with the arguments, the threshold, and per-split
metrics. With answer generation, per-question predictions are written to
`--predictions_log_dir`.
