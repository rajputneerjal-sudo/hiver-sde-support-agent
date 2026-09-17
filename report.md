# Report: AI Support Agent for @AmazonHelp

All numbers below come from an actual run of `eval/run_eval.py` using the real

AmazonHelp data prepared from the Kaggle Customer Support on Twitter dataset.

The evaluation set is a separate 203-example hand-labelled golden set.

Reproduce with:

```bash

python eval/run_eval.py

python eval/judge_agreement.py

```

---

## 1. Problem framing

**Brand chosen:** AmazonHelp.

The goal is to build a practical customer-support agent that can classify an

incoming customer message, retrieve a relevant historical resolution, draft a

grounded response, and decide whether the case should be escalated.

### Intent taxonomy

The agent uses seven intents:

| Intent | Definition |

|---|---|

| `delivery_issue` | Package late/missing, tracking, carrier, or delivery-date problems |

| `refund_return` | Refund, return, money-back, or return-label requests |

| `payment_billing` | Charges, payment, billing, duplicate-charge problems |

| `account_access` | Login, account-access, locked-account, or password problems |

| `product_problem` | Damaged, defective, broken, faulty, or non-working product |

| `order_management` | Cancel, change, reorder, replacement, or order-detail requests |

| `other` | Requests that do not fit the other categories |

The golden set was labelled using the customer's main request as the primary

intent. Ambiguous cases were conservatively assigned to `other`.

### What "good" means

- **Money and account-security issues should not be silently mishandled.**

  The escalation policy therefore treats `payment_billing`, `account_access`,

  and `refund_return` as sensitive cases requiring human review.

- **Grounded replies are preferred over fluent but unsupported replies.**

  Responses are based on retrieved historical AmazonHelp resolutions rather

  than unrestricted generation.

- **The system should outperform simple baselines**, particularly on macro-F1,

  so that performance is not explained only by the dominant `other` class.

### What I chose not to build

- **No multi-turn conversation state.** Each incoming message is processed

  independently.

- **No general sentiment model.** Escalation uses auditable risk/complaint

  trigger phrases instead.

- **No fine-tuned transformer model.** TF-IDF + Logistic Regression was chosen

  because it is fast, inexpensive, interpretable, and easy to reproduce.

- **No automatic sending for escalated cases.** The system produces a draft,

  while escalated cases are intended for human review.

---

## 2. Data and evaluation design

The project uses real AmazonHelp customer-support interactions from the Kaggle

Customer Support on Twitter dataset.

A real customer-to-AmazonHelp response corpus was extracted and used as the

retrieval knowledge base. The processed AmazonHelp knowledge base contains

100,232 customer/response pairs before the golden-set exclusion step.

### Training and golden data

- **800 real labelled examples**: used for training the intent classifier.

- **203 real hand-labelled examples**: held out as the golden evaluation set.

- The golden set was kept separate from training.

- The evaluation retrieval knowledge base excludes customer IDs represented in

  the golden set to reduce direct retrieval leakage.

The golden set is imbalanced:

| Intent | Count |

|---|---:|

| `account_access` | 8 |

| `delivery_issue` | 53 |

| `order_management` | 11 |

| `other` | 106 |

| `payment_billing` | 8 |

| `product_problem` | 10 |

| `refund_return` | 7 |

| **Total** | **203** |

The high proportion of `other` is an important limitation when interpreting

accuracy.

---

## 3. System design

### Intent classifier

The production classifier combines:

1. High-precision rules for obvious account, payment, refund, product, and

   delivery phrases.

2. TF-IDF features with unigram and bigram features.

3. Logistic Regression with balanced class weights.

The classifier is trained only on the 800-example training set.

### Retrieval

The system retrieves the most similar historical AmazonHelp customer/response

examples using TF-IDF cosine similarity.

The evaluation retriever uses `kb_for_eval.csv`, which excludes golden-set

customer IDs.

### Reply generation

The reply generator uses a retrieved historical AmazonHelp response as the

grounding source and records the evidence used for the draft.

This makes the response traceable to an observed support resolution rather than

inventing an answer from scratch.

### Escalation

The current policy escalates:

- `account_access`

- `payment_billing`

- `refund_return`

It also considers serious complaint/legal/fraud language and low retrieval

similarity.

Escalated cases are treated as drafts for human review rather than automatic

customer sends.

---

## 4. Results vs. baselines

Evaluation was performed on the **203-example held-out golden set**.

### Intent classification

| Model | Accuracy | Macro F1 |

|---|---:|---:|

| **Trivial** (majority class) | 0.522 | 0.098 |

| **Simple** (hand-written keyword rules) | 0.527 | 0.285 |

| **System** (rules + TF-IDF + Logistic Regression) | **0.640** | **0.525** |

The system improves over the trivial baseline by **11.8 percentage points in

accuracy** and **0.427 macro-F1**. It also improves over the simple keyword

baseline by **11.3 percentage points in accuracy** and **0.240 macro-F1**.

Macro-F1 is especially informative here because the golden set is strongly

imbalanced.

### Escalation decision

The escalation reference is **policy-derived from the hand-labelled intent

taxonomy**, rather than an independent human-labelled escalation ground

truth. Sensitive intents are treated as escalation-positive.

| Policy | Precision | Recall | F1 |

|---|---:|---:|---:|

| Never escalate | 0.000 | 0.000 | 0.000 |

| Always escalate | 0.113 | 1.000 | 0.204 |

| Simple keyword rule | 0.412 | 0.304 | 0.350 |

| **System policy** | **0.375** | **0.522** | **0.436** |

The system policy has the best F1 of the evaluated escalation policies, but its

recall is only 0.522. Therefore the evaluation does **not** support a claim

that the system reliably catches every case that should reach a human.

### Reply quality

The current headline reply-quality evaluation uses the **offline heuristic

judge**, not a real LLM judge.

| Groundedness | Correctness | Tone | Completeness | Overall |

|---:|---:|---:|---:|---:|

1.24 | 3.71 | 4.00 | 3.44 | **3.10 / 5**

These scores should be interpreted cautiously because the heuristic judge has

known weaknesses, particularly for groundedness.

---

## 5. Failure analysis

The real-data evaluation shows several important failure modes.

### 1. Confusion among `other` and specific intents

The largest class in the golden set is `other` (106/203). Many short or

context-dependent customer tweets do not contain enough explicit information

to map cleanly to the seven-intent taxonomy.

The system therefore sometimes assigns a specific intent to a message that was

labelled `other`, or predicts `other` when the customer request is specific.

**Likely cause:** sparse/ambiguous language combined with a deliberately

coarse taxonomy.

**Improvement:** add more diverse labelled examples, particularly difficult

`other` examples, and consider a confidence threshold that routes uncertain

cases to human review.

### 2. Blended-intent messages

Some messages contain multiple issues, for example a defective product together

with an explicit refund request. TF-IDF classification can focus on the

dominant surface vocabulary instead of the customer's primary requested

resolution.

**Improvement:** add explicit resolution-priority rules, especially for phrases

such as "refund me" and "give me my money back", or use a multi-intent

intermediate representation before selecting the primary intent.

### 3. Short and context-dependent tweets

Tweets can be extremely short and may depend on earlier messages in a thread.

Because this version processes each message independently, the model may not

have enough context to identify the customer's actual request.

**Improvement:** incorporate conversation/thread context and customer history

when available.

### 4. Retrieval can match the wrong sub-scenario

Even when the broad intent is correct, a retrieved response can represent a

different resolution scenario. For example, multiple delivery problems can

share vocabulary while requiring different actions.

**Improvement:** use a second-stage retrieval/reranking step that considers

specific sub-scenario similarity rather than only the broad intent.

### 5. Heuristic reply judge over-rates groundedness

The heuristic groundedness metric is based heavily on overlap with retrieved

material. A response can therefore score well for being grounded in a

historical response even when that historical response is not the right

precedent for the current customer.

This is reflected by the human-agreement result below.

---

## 6. LLM-as-judge and human agreement

Two judge backends are implemented in `eval/llm_judge.py`:

- **Heuristic judge (default/offline):** uses word overlap, entity presence,

  and keyword-based signals. This backend produced the headline reply-quality

  numbers in this report.

- **Real LLM judge:** if `ANTHROPIC_API_KEY` is configured, the same rubric can

  be evaluated using Claude. It was **not used for the headline numbers in this

  report**.

### Human agreement study

A 40-example subset of system outputs was manually rated using the same rubric.

The human reviewer evaluated groundedness, correctness, tone, completeness,

and overall quality.

The agreement script compared those ratings with the heuristic judge:

| Dimension | Pearson r | Within 1 point |

|---|---:|---:|

| Groundedness | undefined | 95% |

| Correctness | 0.15 | 78% |

| Tone | 0.11 | 98% |

| Completeness | 0.38 | 95% |

| **Overall** | **0.42** | **88%** |

Groundedness correlation is undefined because the heuristic judge gave all 40

examples the same groundedness score, producing zero variance.

The overall Pearson correlation of **0.42** indicates only moderate agreement.

Correctness and tone correlations are weak, while completeness is the

strongest of the four dimensions. The 88% within-one-point agreement is useful

but should not be interpreted as strong proof of judge validity.

The result supports treating the heuristic judge as a reproducible screening

metric rather than as a definitive substitute for human review.

---

## 7. What is misleading about the headline numbers?

A headline such as **64.0% intent accuracy** can sound more informative than it

really is. Several factors need to be considered.

1. **The golden set is small and imbalanced.** It contains only 203 examples and

   52.2% belong to `other`. Accuracy can therefore be influenced strongly by

   the class distribution. Macro-F1 provides a more balanced view.

2. **Escalation ground truth is policy-derived.** The escalation reference is

   constructed from the intent labels and the chosen safety policy. It is not

   an independent annotation of what an actual Amazon support team would

   escalate. The escalation numbers therefore measure consistency with the

   designed policy, not real-world escalation correctness.

3. **Reply quality is judged heuristically.** The 3.10/5 score is reproducible

   offline, but the human-agreement study demonstrates that the heuristic has

   a significant groundedness blind spot.

4. **Aggregate metrics hide individual bad replies.** A system can have an

   acceptable average while producing a problematic response for a particular

   customer. Output inspection remains necessary for a support application.

5. **The taxonomy is intentionally small.** Real customer-support workflows

   often contain finer-grained resolution scenarios than seven broad intents.

   A correct broad classification does not guarantee that the retrieved

   resolution is operationally correct.

6. **The system is not production-ready.** It is an offline take-home

   evaluation prototype. It does not have authenticated account access,

   transactional order APIs, human-agent tooling, or multi-turn conversation

   state.

---

## 8. What I would do next with one more week

1. **Expand the real-data labelled set.** Add more examples for the minority

   intents and difficult `other` cases, then retrain and reevaluate.

2. **Improve blended-intent handling.** Add explicit primary-resolution rules

   and regression tests for messages containing multiple issues.

3. **Improve retrieval specificity.** Introduce sub-scenario labels or a

   second-stage reranker so that similar but operationally different cases are

   not treated as interchangeable.

4. **Fix the heuristic judge's groundedness weakness.** Run the LLM judge on a

   larger human-rated sample and compare it against independent human scores.

5. **Add conversation context.** Use the surrounding Twitter thread when

   available so that short follow-up messages can be interpreted correctly.

6. **Validate escalation policy with domain experts.** The current policy is a

   design choice, not evidence of AmazonHelp's actual internal escalation

   criteria.

---

## 9. Reproducibility

Main evaluation:

```bash

python eval/run_eval.py

```

Human/heuristic agreement:

```bash

python eval/judge_agreement.py

```

Tests:

```bash

python -m pytest -q

```

Important evaluation artifacts:

```text

data/processed/training_labeled.csv

data/processed/golden_set.csv

data/processed/amazonhelp_kb.csv

data/processed/kb_for_eval.csv

eval/run_eval.py

eval/system_predictions.csv

eval/judge_agreement.py

reports/eval_results.json

reports/eval_results.md

reports/judge_agreement_results.txt

reports/decision_log.md

```

---

## Decision log

See `reports/decision_log.md` for the non-obvious implementation and

evaluation decisions and their rationale