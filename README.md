# hiver-sde-support-agent/

An AI customer-support agent for **@AmazonHelp** using real customer-support

data from the Kaggle **Customer Support on Twitter** dataset.

The agent:

1. classifies incoming customer messages into 7 support intents,

2. retrieves a relevant historical AmazonHelp resolution,

3. drafts a grounded response using that evidence, and

4. decides whether the case should be escalated to a human, with a reason.

See [`reports/report.md`](reports/report.md) for the complete problem framing,

evaluation results, failure analysis, human-agreement study, limitations, and

next steps.

---

## 1. Data


This project uses real AmazonHelp customer/response interactions extracted from the Kaggle **Customer Support on Twitter** dataset.

The processed retrieval knowledge base contains **100,232 real customer-to-AmazonHelp response pairs** before exclusion of golden-set examples.

### Dataset availability

The original/processed dataset files are **not included in this GitHub repository because some files exceed GitHub's web-upload size limit**.

The following files are used locally by the evaluation pipeline:

data/
└── processed/
    ├── amazonhelp_kb.csv
    ├── kb_for_eval.csv
    └── training_labeled.csv

eval/
└── golden_set.csv

If you are running the project locally, place the provided dataset files in the corresponding `data/processed/` directory before running the evaluation scripts.

The repository contains the complete source code, evaluation logic, tests, reports, and configuration required to reproduce the system once the dataset files are available.

### Evaluation data

- **800 labelled examples** are used to train the intent classifier.
- **203 real, manually labelled examples** form the held-out golden evaluation set.
- The golden set is kept separate from classifier training.
- `kb_for_eval.csv` excludes customer IDs represented in the golden set to reduce direct retrieval leakage.

The golden set is intentionally imbalanced, with `other` as the largest class. For this reason, both accuracy and macro-F1 are reported.
---

## 2. Intent taxonomy

The system uses seven intents:

| Intent | Definition |

|---|---|

| `delivery_issue` | Package late/missing, tracking, carrier, or delivery-date problems |

| `refund_return` | Refund, return, money-back, or return-label requests |

| `payment_billing` | Charges, payment, billing, or duplicate-charge problems |

| `account_access` | Login, account-access, locked-account, or password problems |

| `product_problem` | Damaged, defective, broken, faulty, or non-working product |

| `order_management` | Cancel, change, reorder, replacement, or order-detail requests |

| `other` | Anything that does not fit the other categories |

For ambiguous examples, the customer's main request is used as the primary

intent and unclear cases are conservatively assigned to `other`.

---

## 3. System architecture

```text

                    customer message

                           |

                           v

              +-------------------------+

              | Intent classification   |

              | High-precision rules +   |

              | TF-IDF + Logistic        |

              | Regression               |

              +-------------------------+

                           |

                           v

              +-------------------------+

              | Historical retrieval    |

              | TF-IDF cosine similarity|

              | over AmazonHelp KB       |

              +-------------------------+

                           |

                           v

              +-------------------------+

              | Grounded reply drafting |

              | Top historical AmazonHelp|

              | resolution as evidence  |

              +-------------------------+

                           |

                           v

              +-------------------------+

              | Escalation policy       |

              | Sensitive intents,      |

              | risk/legal language,    |

              | low retrieval similarity|

              +-------------------------+

                           |

                           v

                 response + escalation

```

### Intent classifier

The production classifier combines:

- high-precision rules for obvious account, payment, refund, product, and

  delivery phrases;

- TF-IDF unigram and bigram features;

- Logistic Regression with balanced class weights.

### Retrieval

Historical AmazonHelp customer/response pairs are indexed using TF-IDF.

Cosine similarity is used to retrieve the closest resolved examples.

### Reply drafting

The response is grounded in a retrieved historical AmazonHelp resolution.

The system also records the evidence used for the draft.

### Escalation

The current policy escalates:

- `account_access`

- `payment_billing`

- `refund_return`

It also escalates when:

- retrieval similarity is very low;

- serious legal/fraud/complaint language is detected.

Escalated cases are intended for human review.

---

## 4. Evaluation

Run the complete evaluation with:

```bash

python eval/run_eval.py

```

The current evaluation is performed on **203 held-out golden examples**.

### Intent results

| Model | Accuracy | Macro F1 |

|---|---:|---:|

| Trivial majority baseline | 0.522 | 0.098 |

| Simple keyword baseline | 0.527 | 0.285 |

| **System: rules + TF-IDF + Logistic Regression** | **0.640** | **0.525** |

### Escalation results

| Policy | Precision | Recall | F1 |

|---|---:|---:|---:|

| Never escalate | 0.000 | 0.000 | 0.000 |

| Always escalate | 0.113 | 1.000 | 0.204 |

| Simple keyword policy | 0.412 | 0.304 | 0.350 |

| **System policy** | **0.375** | **0.522** | **0.436** |

The escalation reference is policy-derived from the labelled intents. It is not

an independently human-labelled escalation ground truth, so these metrics

measure consistency with the designed policy rather than real-world Amazon

internal escalation decisions.

### Reply quality

The current offline reply-quality headline uses a heuristic judge:

| Groundedness | Correctness | Tone | Completeness | Overall |

|---:|---:|---:|---:|---:|

| 1.24 | 3.71 | 4.00 | 3.44 | **3.10 / 5** |

A Claude-based judge backend is implemented in `eval/llm_judge.py` and can be

used when `ANTHROPIC_API_KEY` is configured. The headline numbers above were

generated with the offline heuristic judge.

---

## 5. Human agreement study

A 40-example subset was manually rated using the reply-quality rubric.

Run:

```bash

python eval/judge_agreement.py

```

Current agreement with the heuristic judge:

| Dimension | Pearson r | Within 1 point |

|---|---:|---:|

| Groundedness | undefined | 95% |

| Correctness | 0.15 | 78% |

| Tone | 0.11 | 98% |

| Completeness | 0.38 | 95% |

| **Overall** | **0.42** | **88%** |

Groundedness correlation is undefined because the automated judge had zero

variance on that dimension for the 40-example subset.

The overall Pearson correlation of 0.42 indicates moderate agreement, so the

offline heuristic judge should be treated as a reproducible screening metric,

not as a definitive replacement for human review.

---

## 6. Setup

Create and activate a virtual environment:

### Windows PowerShell

```powershell

python -m venv venv

.\venv\Scripts\Activate.ps1

```

### macOS/Linux

```bash

python3 -m venv venv

source venv/bin/activate

```

Install dependencies:

```bash

pip install -r requirements.txt

```

No API key is required for the default offline evaluation.

If `ANTHROPIC_API_KEY` is configured, the optional Claude backend can be used

for LLM-based reply drafting/judging.

---

## 7. Reproduction commands

### Run tests

```bash

python -m pytest -q

```

Expected result:

```text

7 passed

```

### Run evaluation

```bash

python eval/run_eval.py

```

This regenerates:

```text

reports/eval_results.md

reports/eval_results.json

eval/system_predictions.csv

```

### Run human-vs-judge agreement

```bash

python eval/judge_agreement.py

```

### Run the end-to-end demo

```bash

python scripts/run_demo.py

```

The demo runs several sample customer messages through the complete pipeline.

---

## 8. Important files

```text

hiver-sde-support-agent/

│

├── src/

│   ├── intents.py              # Intent taxonomy + classifier

│   ├── retrieval.py            # TF-IDF historical retrieval

│   ├── reply_generator.py      # Grounded response drafting

│   ├── escalation.py           # Escalation policy

│   ├── pipeline.py             # End-to-end SupportAgent

│   └── llm_backend.py          # Optional Claude integration

│

├── data/

│   └── processed/

│       ├── amazonhelp_kb.csv   # Real AmazonHelp retrieval KB

│       ├── kb_for_eval.csv     # KB with golden examples excluded

│       ├── training_labeled.csv# 800 labelled training examples

│       └── golden_set.csv      # 203 held-out golden examples

│

├── eval/

│   ├── run_eval.py             # Full evaluation

│   ├── metrics.py              # Evaluation metrics

│   ├── llm_judge.py            # Reply-quality judge

│   ├── judge_agreement.py      # Human-vs-judge agreement

│   └── system_predictions.csv  # Per-example evaluation output

│

├── scripts/

│   └── run_demo.py             # End-to-end demo

│

├── reports/

│   ├── report.md               # Main assignment report

│   ├── eval_results.md         # Latest evaluation summary

│   ├── eval_results.json       # Machine-readable results

│   ├── judge_agreement_results.txt

│   └── decision_log.md         # Design/evaluation decisions

│

├── tests/

│   └── test_pipeline.py        # Pipeline sanity tests

│

├── requirements.txt

└── README.md

```

---

## 9. Limitations

This is an offline take-home prototype rather than a production support

system.

Important limitations include:

- only seven broad intents are modelled;

- the golden set contains only 203 examples and is highly imbalanced;

- short tweets may require conversation/thread context;

- retrieval can select a historically similar but operationally different

  resolution;

- escalation ground truth is policy-derived rather than independently

  annotated;

- the headline reply-quality judge is heuristic unless the optional Claude

  backend is configured;

- there is no authenticated access to customer accounts or order systems;

- escalated cases require human review.

See `reports/report.md` for the detailed failure analysis and the "what's

misleading about the headline number" discussion.

---

## 10. Project decisions

Key implementation and evaluation decisions are documented in:

```text

reports/decision_log.md

```

The report explains why the project uses a small intent taxonomy, a

TF-IDF/Logistic Regression classifier, historical retrieval, policy-based

escalation, and separate golden-set evaluation.
