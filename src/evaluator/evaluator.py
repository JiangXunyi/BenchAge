from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import re
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set
import os, re, time, numpy as np, pandas as pd
from openai import OpenAI
import tiktoken 
import pickle 
import csv
import tqdm
from sklearn.metrics import cohen_kappa_score                       
from functools import partial
import concurrent.futures

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ENC = tiktoken.encoding_for_model("gpt-4o-mini")

# ------------------------------------------------------------------
# 0. parameters you might tune
MAX_TOKENS_PER_PROMPT = 30000           # safety margin (< 8k for gpt-4o-mini)
MAX_TOKENS_PER_ANSWER = 10000           # clip each answer individually
ENGINE = "gpt-4o-mini"                  # judge model
# ------------------------------------------------------------------

DATASETS = {"NaturalQuestion", "TruthfulQA", "BoolQ", "FreshQA", "SelfAware", "TriviaQA"}

def _resolve_path(dataset: str, resume_dir: str) -> Dict[str, Path]:
    """return paths we need to use"""
    # 1. web-search jsonl
    base_ts  = Path("data") / dataset / "time-sensitive-data"
    web_dir  = sorted([p for p in base_ts.iterdir()
                       if p.is_dir() and p.name.startswith("websearch")])[-1] # the most recent one
    src_ts   = web_dir / f"{dataset.lower()}-websearch.jsonl"
    if not src_ts.exists():
        raise SystemExit(f"Dataset file not found: {src_ts}")

    # 2. opencompass prediction
    # TODO: Configure your model output path in environment variable or config file
    # Example: export MODEL_OUTPUT_BASE="/path/to/your/model/outputs"
    model_output_base = os.getenv("MODEL_OUTPUT_BASE")
    if not model_output_base:
        raise SystemExit("Please set MODEL_OUTPUT_BASE environment variable to your model output directory")
    
    base_all = Path(model_output_base) / dataset.lower()
    pred_dir = sorted(base_all.iterdir())[-1] / "predictions"
    src_models = sorted([p / f"{dataset.lower()}.json" for p in pred_dir.iterdir()])

    # 3. output path
    if resume_dir:
        out_dir = Path(resume_dir).expanduser()
        if not out_dir.exists():
            raise SystemExit(f"{out_dir} not found")

        return dict(src_ts=src_ts, src_models=src_models, out_dir=out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data") / "Evaluation" / dataset.lower() / ts
        out_dir.mkdir(parents=True, exist_ok=True)

    return dict(src_ts=src_ts, src_models=src_models, out_dir=out_dir)
    
    
def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


_question_cache: Dict[tuple[str, str], str] = {}   # (dataset, raw) -> question

def process_question(dataset: str, prompt_raw: str) -> str:
    """
    process the origin_prompt into question to match with the keys
    """
    key = (dataset, prompt_raw)
    if key in _question_cache:
        return _question_cache[key]

    question = prompt_raw  # By default, return the original string directly

    if dataset == "BoolQ":
        m = re.search(r"Question:\s*(.*?)\s*(?:A\.)", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()

    elif dataset == "FreshQA":
        m = re.match(r"Q:\s*(.*?)\s*A:", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()

    elif dataset == "HaluEval":
        m = re.match(r"Q:\s*(.*?)\s*A:", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()

    elif dataset == "NaturalQuestion":
        m = re.search(r"Question:\s*(.*?)\s*Answer:", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()

    elif dataset == "SelfAware":
        m = re.match(r"Q:\s*(.*?)\s*A:", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()

    elif dataset == "TriviaQA":
        m = re.search(r"Question:\s*(.*?)\s*Answer:", prompt_raw, re.S | re.I)
        if m: question = m.group(1).strip()


    _question_cache[key] = question
    return question


def format_prompt(q, a1, a2):
    return f"""
If Answer 1 and Answer 2 state the same fact (allowing synonyms,
abbreviations, or different date formats), reply **True**.
Otherwise reply **False**.

Reply with exactly one word: True or False.

Q: {q}
Answer 1: {a1}
Answer 2: {a2}
"""

def format_prompt_boolq(q: str, a1: str, a2: str) -> str:
    return f"""
Mapping of options:
A. Yes
B. No

**Step 1 – Normalise each answer**
• If the answer **starts with “A”** (e.g. “A”, “A.”, “A:”) → normalise to **Yes**.  
• If the answer **starts with “B”** (e.g. “B”, “B.”, “B:”) → normalise to **No**.  
• If the answer explicitly contains the word **Yes** → Yes.  
• If the answer explicitly contains the word **No**  → No.  
• Otherwise, read the text and decide whether it asserts Yes or No.

**Step 2 – Consistency decision**  
If the two normalised answers are identical (both Yes or both No), output **True**.  
Otherwise output **False**.

Reply with **exactly one word**: True or False.

Q: {q}
Answer 1: {a1}
Answer 2: {a2}
"""

def to_text(x) -> str:
    """
    Robustly convert a value to plain string.
    • None / NaN        -> ""
    • list / tuple      -> join with ", "
    • everything else   -> str(x)
    """
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, (list, tuple)):
        return ", ".join(map(str, x))
    return str(x)

def truncate_to_tokens(text: str, limit: int) -> str:
    """Keep only first `limit` tokens (roughly) to avoid over-length prompts."""
    text = to_text(text)
    tokens = ENC.encode(text)
    if len(tokens) <= limit:
        return text
    return ENC.decode(tokens[:limit])

def judge_pair(question: str, ans1: str, ans2: str, dataset: str) -> float:
    """GPT judge pairs 1/0/np.nan."""
    if dataset == "BoolQ":
        prompt = format_prompt_boolq(
            question,
            ans1,
            ans2
        )
        system_prompt = "You are a factual-consistency judge for **Yes/No** questions."
    else:
        prompt = format_prompt(
            truncate_to_tokens(question, 256),
            truncate_to_tokens(ans1,     MAX_TOKENS_PER_ANSWER),
            truncate_to_tokens(ans2,     MAX_TOKENS_PER_ANSWER),
        )
        system_prompt = "You are a factual-consistency judge."

    try:
        rsp = client.chat.completions.create(
            model=ENGINE,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": prompt}],
            temperature=0,
            max_tokens=1,
        )
        label_str = rsp.choices[0].message.content.strip().lower()
        score = 1.0 if label_str.startswith("true") else 0.0
    except Exception as e:
        logging.warning("API error -> %s", e)
        label_str, score = "ERROR", np.nan

    return {                       # Always return a dict
        "prompt": prompt,
        "llm_output": label_str,
        "result": score,
    }


def evaluate_pairs(dataset: str,
                   ts_data: dict, non_ts_data: dict,
                   models: list[str], chk_path: Path,
                   max_workers: int = 16) -> dict:
    """
    Parallel evaluation of model pairs.
    results["ts"][model]["llm_vs_gold"] → list[dict]:
        {
          "q": question,
          "prompt": …,
          "llm_output": …,
          "result": True/False
        }
    """
    # checkpoint
    if chk_path.exists():
        with open(chk_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = {"ts": {}, "non": {}}

    def _add(bucket: dict, model: str, key: str, item: dict):
        bucket.setdefault(model, {}).setdefault(key, []).append(item)

    def _evaluate_model_for_question(args):
        """Helper function for parallel processing"""
        q, info, model, dataset_name, is_ts = args
        
        if model not in info:
            return None
            
        ans = info[model]
        gold = ", ".join(info["gold"]) if isinstance(info["gold"], list) else str(info["gold"])
        
        model_results = []
        
        if is_ts:
            search = str(info["search"])
            
            # LLM vs Gold
            jud_lg = judge_pair(q, gold, ans, dataset_name)
            model_results.append(("llm_vs_gold", {"question": q, **jud_lg}))
            
            # LLM vs Search
            jud_ls = judge_pair(q, search, ans, dataset_name)
            model_results.append(("llm_vs_search", {"question": q, **jud_ls}))
            
            # Gold vs Search (only compute once per question)
            if "gs" not in info:
                info["gs"] = judge_pair(q, gold, search, dataset_name)
            model_results.append(("gold_vs_search", {"question": q, **info["gs"]}))
        else:
            # Non-TS: only LLM vs Gold
            jud = judge_pair(q, gold, ans, dataset_name)
            model_results.append(("llm_vs_gold", {"question": q, **jud}))
        
        return (model, model_results)

    # Process time-sensitive questions
    if ts_data:
        logging.info("Processing TS questions with %d workers", max_workers)
        
        # Prepare args for parallel processing
        ts_args = []
        for q, info in ts_data.items():
            for m in models:
                ts_args.append((q, info, m, dataset, True))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(_evaluate_model_for_question, args): args 
                             for args in ts_args}
            
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_args), 
                                   total=len(ts_args), desc="TS evaluation"):
                try:
                    result = future.result()
                    if result is not None:
                        model, model_results = result
                        for key, item in model_results:
                            _add(results["ts"], model, key, item)
                except Exception as e:
                    logging.warning("Error processing TS question: %s", e)
        
        # Save checkpoint after TS
        with open(chk_path, "wb") as f:
            pickle.dump(results, f)

    # Process non-TS questions
    if non_ts_data:
        logging.info("Processing non-TS questions with %d workers", max_workers)
        
        # Prepare args for parallel processing
        non_ts_args = []
        for q, info in non_ts_data.items():
            for m in models:
                non_ts_args.append((q, info, m, dataset, False))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(_evaluate_model_for_question, args): args 
                             for args in non_ts_args}
            
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_args), 
                                   total=len(non_ts_args), desc="Non-TS evaluation"):
                try:
                    result = future.result()
                    if result is not None:
                        model, model_results = result
                        for key, item in model_results:
                            _add(results["non"], model, key, item)
                except Exception as e:
                    logging.warning("Error processing non-TS question: %s", e)

    # Final checkpoint
    with open(chk_path, "wb") as f:
        pickle.dump(results, f)

    return results

def _update_csv(path: Path, row: dict[str, float], dataset: str) -> None:
    """
    Row = model, Column = dataset
    If the file exists: append a new column or update an existing column;
    If not: create header ["model", <dataset>]
    """
    # ── initialize table ───────────────────────────────
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, newline="") as f:
            table = list(csv.reader(f))
        header = table[0][1:]                      # current dataset
        if dataset not in header:                  # add new dataset into column
            header.append(dataset)
            table[0] = ["model", *header]
            for r in table[1:]:
                r.extend([""] * (len(header) - (len(r) - 1)))
    else:
        header = [dataset]
        table  = [["model", *header]]

    col_idx = header.index(dataset) + 1            # column index

    # ── Update each model in row into the table ───────────────
    model_map = {r[0]: r for r in table[1:]}       # Existing row index

    for model, val in row.items():
        if model in model_map:                     # Update existing row
            r = model_map[model]
            while len(r) <= col_idx:
                r.append("")                       # Fill columns
            r[col_idx] = val
        else:                                      # Add new row
            new_row = [model] + [""] * len(header)
            new_row[col_idx] = val
            table.append(new_row)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(table)

def summarise_and_write(dataset: str, results: dict, out_root: Path) -> None:
    acc_diff_row, kappa_row, overall_row = {}, {}, {}

    # —— Utility: convert dict/float list to numeric list ——
    def _to_scores(lst):
        return [
            (d["result"] if isinstance(d, dict) else d)
            for d in lst
        ]

    for model in results["ts"].keys() | results["non"].keys():
        # ---------- non-TS ----------
        rec_non   = results["non"].get(model, {})
        scores_non = _to_scores(rec_non.get("llm_vs_gold", []))
        n_non      = len(scores_non)
        correct_non = np.nansum(scores_non)
        acc_non    = correct_non / n_non if n_non else np.nan

        # ---------- TS ----------
        rec_ts   = results["ts"].get(model, {})
        scores_lg = _to_scores(rec_ts.get("llm_vs_gold",   []))
        scores_ls = _to_scores(rec_ts.get("llm_vs_search", []))
        scores_gs = _to_scores(rec_ts.get("gold_vs_search", []))

        n_ts          = len(scores_lg)
        correct_ts    = np.nansum(scores_lg)
        correct_search = np.nansum(scores_ls)

        acc_ts        = correct_ts    / n_ts if n_ts else np.nan
        acc_search_ts = correct_search / n_ts if n_ts else np.nan

        # ---------- overall ----------
        weighted_sum = correct_non + correct_ts
        total_q      = n_non + n_ts
        acc_overall  = weighted_sum / total_q if total_q else np.nan

        # ---------- κ ----------
        if n_ts:
            k1 = cohen_kappa_score(scores_lg, scores_ls)
            k2 = cohen_kappa_score(scores_gs, scores_lg)
            k3 = cohen_kappa_score(scores_gs, scores_ls)
            kappa_avg = np.nanmean([k1, k2, k3])
        else:
            kappa_avg = np.nan

    # ---------- Save ----------
        acc_diff_row[model] = acc_search_ts - acc_ts
        kappa_row[model]    = kappa_avg
        overall_row[model]  = acc_overall

    # —— 写出 CSV ——————————————————————————
    out_root = out_root / "Summary"
    _update_csv(out_root / "accuracy_diff.csv", acc_diff_row, dataset)
    _update_csv(out_root / "avg_kappa.csv",     kappa_row,    dataset)
    _update_csv(out_root / "overall_acc.csv",   overall_row,  dataset)



def _read_json_or_jsonl(path: Path):
    """ return list[Dict] for .json / .jsonl"""
    if path.suffix == ".jsonl":
        with path.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with path.open("r") as f:
            return json.load(f)

def _extract_prompt(raw):
    """
    Supports three formats:
      1. list[{"role": "HUMAN", "prompt": "..."}]
      2. {"role": "HUMAN", "prompt": "..."}      # Rare single dict
      3. str                                     # Already expanded
    """
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0].get("prompt", "")
    if isinstance(raw, dict):
        return raw.get("prompt", "")
    return str(raw)  # fallback


def build_q_dicts(
    dataset: str,
    src_ts: Path,
    src_models: list[Path],
) -> tuple[dict, dict]:
    """
    return (ts_data, non_ts_data):
        ts_data[q]   = {"gold": str,
                        "search": str,
                        "<model1>": pred, ...}
        non_ts_data[q] = {"gold": str,
                          "<model1>": pred, ...}
    """
    #  time-sensitive search results
    ts_raw = _read_json_or_jsonl(src_ts)
    if dataset == "NaturalQuestion":
        ts_data = {
            d["question"].strip() + "?" : {"search": d["search_answer"]}
            for d in ts_raw
        }
    else:
        ts_data = {
            d["question"].strip(): {"search": d["search_answer"]}
            for d in ts_raw
        }
    non_ts_data: dict[str, dict] = {}

    # get all open-source model answers
    for mpath in src_models:
        model_name = mpath.parent.name
        with open(mpath, "r") as f:
            samples = json.load(f) # dictionary

        for _, s in tqdm.tqdm(samples.items()):
            gold_raw = s.get("gold")
            if gold_raw is None and dataset == "SelfAware":
                prompt_raw = _extract_prompt(s["origin_prompt"])
                q = process_question(dataset, prompt_raw)
                ts_data.pop(q, None)
                continue
            
            prompt_raw = _extract_prompt(s["origin_prompt"])
            q = process_question(dataset, prompt_raw)
            tgt = ts_data if q in ts_data else non_ts_data
            entry = tgt.setdefault(q, {})
            entry[model_name] = s["prediction"]
            entry["gold"] = gold_raw
            

    return ts_data, non_ts_data


def main(argv: list[str] | None = None) -> None:
    """CLI entry: call each sub-step to complete a dataset evaluation."""
    prs = argparse.ArgumentParser(
        description="Evaluate accuracy-diff & Cohen’s κ on time-sensitive benchmarks"
    )
    prs.add_argument("dataset", choices=sorted(DATASETS),
                     help="Target dataset to evaluate")
    prs.add_argument("--resume_dir", type=str,)
    args = prs.parse_args(argv)

    # get all paths
    paths = _resolve_path(args.dataset, args.resume_dir)

    # start logging data/Evaluation/<dataset>/<timestamp>/log/run.log
    _setup_logging(paths["out_dir"] / "log" / "run.log")
    logging.info("Start evaluation | dataset=%s", args.dataset)

    # get questions and answers
    ts_data, non_ts_data = build_q_dicts(
        dataset     = args.dataset,
        src_ts      = paths["src_ts"],
        src_models  = paths["src_models"]
    )
    logging.info("Loaded %d TS questions, %d non-TS questions",
                 len(ts_data), len(non_ts_data))

    # llm-as-jjudge
    models = [p.parent.name for p in paths["src_models"]]
    chk_path = paths["out_dir"] / "checkpoint.pkl"
    results  = evaluate_pairs(args.dataset, ts_data, non_ts_data,
                              models, chk_path)
    logging.info("Pairwise evaluation finished")

    full_path = paths["out_dir"] / "results_full.pkl"
    with open(full_path, "wb") as f:
        pickle.dump(results, f)
    logging.info("Saved full result to %s", full_path)

    summarise_and_write(args.dataset, results, paths["out_dir"].parent)
    logging.info("Updated accuracy_diff.csv & avg_kappa.csv")

    logging.info("All done")


if __name__ == "__main__":
    main()


