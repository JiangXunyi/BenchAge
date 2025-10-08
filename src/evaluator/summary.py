import argparse
import pickle
import numpy as np
import csv
import os
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

def update_csv(path: Path, row: dict[str, float], dataset: str) -> None:
    """Write (or update) the row into the CSV with the same name: row=model, column=dataset"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, newline="") as f:
            table = list(csv.reader(f))
        header = table[0][1:]             # existing dataset columns
        if dataset not in header:
            header.append(dataset)
            table[0] = ["model", *header]
            for r in table[1:]:
                r.extend([""] * (len(header) - (len(r) - 1)))
    else:
        header = [dataset]
        table  = [["model", *header]]

    col_idx = header.index(dataset) + 1
    model_map = {r[0]: r for r in table[1:]}

    for model, val in row.items():
        if model in model_map:
            row_cells = model_map[model]
            while len(row_cells) <= col_idx:
                row_cells.append("")
            row_cells[col_idx] = val
        else:
            new_row = [model] + [""] * len(header)
            new_row[col_idx] = val
            table.append(new_row)

    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(table)

def summarise(dataset: str, results: dict, out_root: Path) -> None:
    acc_diff_row, kappa_row, overall_row = {}, {}, {}

    for model in results["ts"].keys() | results["non"].keys():
        # ---------- non-TS ----------
        rec_non = results["non"].get(model, {})
        n_non   = len(rec_non.get("llm_vs_gold", []))
        correct_non = np.nansum(rec_non.get("llm_vs_gold", []))
        acc_non = correct_non / n_non if n_non else np.nan

        # ---------- TS ----------
        rec_ts = results["ts"].get(model, {})
        n_ts   = len(rec_ts.get("llm_vs_gold", []))
        correct_search = np.nansum(rec_ts.get("llm_vs_search", []))
        correct_ts     = np.nansum(rec_ts.get("llm_vs_gold", []))
        acc_search_ts  = correct_search / n_ts if n_ts else np.nan
        acc_ts         = correct_ts / n_ts if n_ts else np.nan

        # ---------- overall ----------
        total_q     = n_non + n_ts
        weighted_ok = correct_non + correct_ts
        acc_overall = weighted_ok / total_q if total_q else np.nan

        # ---------- kappa ----------
        if n_ts:
            k1 = cohen_kappa_score(rec_ts["llm_vs_gold"],    rec_ts["llm_vs_search"])
            k2 = cohen_kappa_score(rec_ts["gold_vs_search"], rec_ts["llm_vs_gold"])
            k3 = cohen_kappa_score(rec_ts["gold_vs_search"], rec_ts["llm_vs_search"])
            kappa_avg = np.nanmean([k1, k2, k3])
        else:
            kappa_avg = np.nan

        # ---------- collect ----------
        acc_diff_row[model] = acc_search_ts - acc_ts   # corrected diff
        kappa_row[model]    = kappa_avg
        overall_row[model]  = acc_overall

    # ---------- Write three tables ----------
    summary_dir = out_root / "Summary"
    update_csv(summary_dir / "accuracy_diff_ts.csv", acc_diff_row, dataset)
    # update_csv(summary_dir / "avg_kappa.csv",     kappa_row,    dataset)
    # update_csv(summary_dir / "overall_acc.csv",   overall_row,  dataset)


def main():
    parser = argparse.ArgumentParser(description="Re-compute accuracy diff for an existing dataset")
    parser.add_argument("dataset", help="Dataset name, e.g. NaturalQuestion")
    args = parser.parse_args()

    ds = args.dataset.strip()
    eval_dir = Path("data") / "Evaluation" / ds.lower()
    

    # folder
    time = os.listdir(eval_dir)[0]
    print(time)

    # Allow two possible filenames
    pkl_path = next((p for p in [eval_dir / time / "results_all.pkl",
                                 eval_dir / time / "results_full.pkl"] if p.exists()), None)
    if pkl_path is None:
        raise SystemExit(f"✘ Result file not found: {eval_dir}/results_all.pkl or results_full.pkl")

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    
        summarise(ds, results, eval_dir)
    print(f"New {eval_dir/'Summary/accuracy_diff_ts.csv'}")


if __name__ == "__main__":
    main()