# -*- coding: utf-8 -*-
"""websearch.py ‒ Dataset-aware, resumable web search runner

```
data/<ds>/time-sensitive-data/websearch-<YYYYMMDD_HHMMSS>/
│   <ds>-websearch.jsonl     ← streaming results (full payload)
│   checkpoint.ids           ← processed question IDs for resumption
├── log/
│   run.log                  ← combined log/stdout
└── stats.txt
```
Usage examples
--------------
```bash
python websearch.py TruthfulQA --model qwen-14b
python websearch.py NaturalQuestion -e brave google --resume
```

Command-line flags
------------------
* ``dataset`` (positional): one of ``DATASETS``.
* ``--model``: optional model identifier (stored in stats only).
* ``--resume``: continue from existing checkpoint if present.
* ``--max`` / ``--timeout`` / ``--engines`` mirror previous script.

Dependencies: stdlib + pyyaml + ``brave_search`` / ``ReAct_google`` modules.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set

from src.runners import brave_search
from src.runners import ReAct_google

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

###############################################################################
# Constants                                                                    
###############################################################################
DATASETS = {"NaturalQuestion", "TruthfulQA", "BoolQ", "FreshQA", "HaluEval", "SelfAware", "TriviaQA"}

###############################################################################
# Engine interface & registry          
###############################################################################
class SearchEngine:
    name: str

    def answer(self, question: str, timeout: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class BraveWikipediaEngine(SearchEngine):
    name = "brave"

    def __init__(self) -> None:
        self._impl = brave_search

    def answer(self, question: str, timeout: int) -> Optional[Dict[str, Any]]:  # noqa: D401
        return self._impl.search_wikipedia(question, timeout=timeout)  # type: ignore[arg-type]


class GoogleReActEngine(SearchEngine):
    name = "google"

    def __init__(self) -> None:
        self._impl = ReAct_google

    def answer(self, question: str, timeout: int) -> Optional[Dict[str, Any]]:  # noqa: D401
        fn = getattr(self._impl, "main", None) or self._impl.google_qa  # type: ignore[attr-defined]
        return fn(question, timeout=timeout)  # type: ignore[arg-type]


ENGINE_REGISTRY = {"brave": BraveWikipediaEngine, "google": GoogleReActEngine}



def _iter_jsonl(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _resolve_paths(dataset: str) -> Dict[str, pathlib.Path]:
    base = pathlib.Path("data") / dataset / "time-sensitive-data"
    src = base / f"{dataset.lower()}-time-sensitive.jsonl"
    if not src.exists():
        raise SystemExit(f"Dataset file not found: {src}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"websearch-{ts}"
    (run_dir / "log").mkdir(parents=True, exist_ok=True)

    return {
        "src": src,
        "run_dir": run_dir,
        "results": run_dir / f"{dataset.lower()}-websearch.jsonl",
        "results_metadata": run_dir / f"{dataset.lower()}-websearch-metadata.jsonl",
        "checkpoint": run_dir / "checkpoint.ids",
        "stats": run_dir / "stats.txt",
        "log": run_dir / "log" / "run.log",
    }


def _setup_logging(log_path: pathlib.Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )

###############################################################################
# Main                                                                         
###############################################################################

def main(argv: List[str] | None = None) -> None:
    prs = argparse.ArgumentParser(description="Dataset-aware web search runner")
    prs.add_argument("dataset", choices=sorted(DATASETS))
    prs.add_argument("--model", help="Model identifier (stored in stats)")
    prs.add_argument("--engines", "-e", nargs="*", choices=list(ENGINE_REGISTRY), default=["brave", "google"])
    prs.add_argument("--timeout", type=int, default=10)
    prs.add_argument("--max", type=int)
    prs.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if present")
    args = prs.parse_args(argv)

    paths = _resolve_paths(args.dataset)
    _setup_logging(paths["log"])

    logging.info("Run dir: %s", paths["run_dir"])
    logging.info("Engines: %s", args.engines)

    # Build engines
    engines: List[SearchEngine] = [ENGINE_REGISTRY[n]() for n in args.engines]
    # Write model into os.environ for use in engines
    if args.model:
        os.environ["OPENAI_DEFAULT_MODEL"] = args.model
        logging.info("Model set to: %s", args.model)

    # Determine processed IDs if resuming
    processed: Set[int] = set()
    if args.resume and paths["checkpoint"].exists():
        processed = {int(line.strip()) for line in paths["checkpoint"].read_text().splitlines() if line.strip()}
        logging.info("Resuming – %d questions already done", len(processed))

    src_iter = _iter_jsonl(paths["src"])

    # Open files line-buffered
    results_fh = paths["results"].open("a", encoding="utf-8", buffering=1)
    results_metadata_fh = paths["results_metadata"].open("a", encoding="utf-8", buffering=1)
    ckpt_fh = paths["checkpoint"].open("a", encoding="utf-8", buffering=1)

    n_total = n_done = 0
    start_time = time.time()

    try:
        for rec in src_iter:
            qid = rec.get("id") or rec.get("question_id") or hash(rec["question"])
            if qid in processed:
                continue
            n_total += 1
            if args.max and n_total > args.max:
                break
            question = rec["question"]
            logging.info("[%d] %s", int(qid), question[:80])

            status = "unanswered"
            payload: Dict[str, Any] | None = None

            for eng in engines:
                try:
                    payload = eng.answer(question, timeout=args.timeout)
                    if payload and payload.get("search_answer") == "Not found":
                        logging.info("Engine %s did not find an answer, switch to google search ", eng.name)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logging.exception("Engine %s failed: %s", eng.name, exc)
                    continue
                if payload:
                    status = eng.name
                    break


            json.dump({"id": qid, "question": question, "status": status, "payload": payload}, results_metadata_fh, ensure_ascii=False)
            results_metadata_fh.write("\n")
            results_metadata_fh.flush()
            
            if payload:
                json.dump({"id": qid, "question": question, "status": status, "search_answer": payload.get("search_answer", ""), "urls": payload.get("urls", [])}, results_fh, ensure_ascii=False)
            else:
                json.dump({"id": qid, "question": question, "status": status, "search_answer": "", "urls": []}, results_fh, ensure_ascii=False)
            results_fh.write("\n")
            results_fh.flush()

            ckpt_fh.write(f"{qid}\n")
            ckpt_fh.flush()

            processed.add(qid)
            n_done += 1
    finally:
        results_fh.close()
        ckpt_fh.close()

    duration = time.time() - start_time
    with paths["stats"].open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "dataset": args.dataset,
            "model": args.model,
            "total_processed": n_done,
            "total_skipped": len(processed) - n_done,
            "runtime_sec": round(duration, 2),
            "engines": args.engines,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))

    logging.info("Finished – processed %d questions in %.1fs", n_done, duration)


if __name__ == "__main__":
    main()