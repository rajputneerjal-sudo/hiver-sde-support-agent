"""
Compares the offline heuristic judge's scores against the hand-labeled
human scores in human_labels.py, on the same 40 examples in
judge_subset_raw.csv. Reports per-dimension and overall correlation plus a
plain "within 1 point" agreement rate, since Pearson r alone can look
misleadingly strong or weak on a 1-5 scale with low variance.

Run: python3 eval/judge_agreement.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.llm_judge import heuristic_score  # noqa
from eval.human_labels import HUMAN_SCORES  # noqa

RAW_PATH = Path(__file__).resolve().parent / "judge_subset_raw.csv"
DIMENSIONS = ["groundedness", "correctness", "tone", "completeness"]


def main():
    rows = list(csv.DictReader(open(RAW_PATH)))

    human_vals = {d: [] for d in DIMENSIONS}
    judge_vals = {d: [] for d in DIMENSIONS}
    human_overall, judge_overall = [], []

    matched = 0
    for r in rows:
        rid = int(r["id"])
        if rid not in HUMAN_SCORES:
            continue
        matched += 1
        h_g, h_c, h_t, h_comp = HUMAN_SCORES[rid]
        js = heuristic_score(
            message=r["text"], reply=r["reply"],
            evidence=[r["evidence"]] if r["evidence"] else [],
            escalate=(r["escalate"] == "True"),
        )
        human_vals["groundedness"].append(h_g); judge_vals["groundedness"].append(js.groundedness)
        human_vals["correctness"].append(h_c); judge_vals["correctness"].append(js.correctness)
        human_vals["tone"].append(h_t); judge_vals["tone"].append(js.tone)
        human_vals["completeness"].append(h_comp); judge_vals["completeness"].append(js.completeness)
        human_overall.append((h_g + h_c + h_t + h_comp) / 4)
        judge_overall.append(js.overall)

    print(f"Matched {matched} examples between raw subset and human labels.\n")
    print(f"{'dimension':<14}{'pearson_r':>10}{'spearman_r':>12}{'within_1pt':>12}")
    for d in DIMENSIONS:
        h, j = np.array(human_vals[d]), np.array(judge_vals[d])
        if np.std(h) == 0 or np.std(j) == 0:
            r_p, r_s = float("nan"), float("nan")
        else:
            r_p, _ = pearsonr(h, j)
            r_s, _ = spearmanr(h, j)
        within1 = float(np.mean(np.abs(h - j) <= 1))
        print(f"{d:<14}{r_p:>10.2f}{r_s:>12.2f}{within1:>12.0%}")

    h_o, j_o = np.array(human_overall), np.array(judge_overall)
    r_p, _ = pearsonr(h_o, j_o)
    within1_o = float(np.mean(np.abs(h_o - j_o) <= 1))
    print(f"\noverall score: pearson_r={r_p:.2f}, within_1pt_agreement={within1_o:.0%}")
    print(f"mean human overall={h_o.mean():.2f}, mean judge overall={j_o.mean():.2f}")

    out_path = Path(__file__).resolve().parent.parent / "reports" / "judge_agreement_results.txt"
    with open(out_path, "w") as f:
        f.write(f"Matched {matched} examples.\n")
        f.write(f"Overall Pearson r = {r_p:.2f}\n")
        f.write(f"Overall within-1-point agreement = {within1_o:.0%}\n")
        f.write(f"Mean human overall = {h_o.mean():.2f}, mean judge overall = {j_o.mean():.2f}\n")
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
