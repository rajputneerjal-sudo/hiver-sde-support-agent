# Evaluation Results
Golden set size: **203 real hand-labelled examples**.
The classifier was trained on the separate `training_labeled.csv` training set. The evaluation retriever used `kb_for_eval.csv`, which excludes golden-set customer IDs.
## Intent classification
| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Trivial (majority class) | 0.522 | 0.098 |
| Simple (keyword rules) | 0.527 | 0.285 |
| System (TF-IDF + LogReg) | 0.640 | 0.525 |

## Escalation decision
| Policy | Precision | Recall | F1 | Escalation rate (true / pred) |
|---|---:|---:|---:|---:|
| Never escalate | 0.000 | 0.000 | 0.000 | 0.11 / 0.00 |
| Always escalate | 0.113 | 1.000 | 0.204 | 0.11 / 1.00 |
| Simple keyword rule | 0.412 | 0.304 | 0.350 | 0.11 / 0.08 |
| System policy | 0.375 | 0.522 | 0.436 | 0.11 / 0.16 |

## Reply quality
Judge backend: `heuristic`
| Groundedness | Correctness | Tone | Completeness | Overall |
|---:|---:|---:|---:|---:|
| 1.24 | 3.71 | 4.0 | 3.44 | 3.1 |
