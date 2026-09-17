# Decision Log

This document records the main non-obvious implementation and evaluation
decisions for the Hiver SDE Intern take-home project.

## 1. Chose AmazonHelp as the brand

AmazonHelp was selected because the Customer Support on Twitter dataset contains
a large number of Amazon customer-support interactions. This provides enough
variety to define a small but useful e-commerce support taxonomy.

## 2. Used real AmazonHelp customer/response data

The final project uses real AmazonHelp customer-to-response pairs extracted from
the Kaggle Customer Support on Twitter dataset.

The processed retrieval knowledge base contains 100,232 customer/response pairs
before golden-set exclusion.

This replaced the original synthetic-data approach used during early project
development. The synthetic generator is not used for the final evaluation.

## 3. Chose seven intents

The final taxonomy is intentionally small:

- `delivery_issue`
- `refund_return`
- `payment_billing`
- `account_access`
- `product_problem`
- `order_management`
- `other`

The categories were chosen to represent recurring e-commerce support problems
without creating many sparse classes.

`other` is intentionally broad. The golden set contains many ambiguous or
context-dependent messages, so forcing every message into a highly specific
category would create artificial precision.

## 4. Used a separate held-out golden set

A 203-example golden set was separated from the training data and manually
labelled.

The training set contains 800 labelled examples.

The golden set is not used to train the classifier. This separation makes the
reported intent metrics a held-out evaluation rather than training-set
performance.

## 5. Kept the golden set out of the evaluation retrieval KB

The project creates `data/processed/kb_for_eval.csv` by removing customer IDs
represented in the golden set from the historical retrieval corpus.

The original KB contains 100,232 rows. The evaluation KB contains 100,013 rows,
with 219 rows removed.

This reduces the risk that the retriever simply finds the exact customer
example being evaluated.

## 6. Used TF-IDF + Logistic Regression for intent classification

The classifier uses TF-IDF unigram/bigram features and Logistic Regression with
balanced class weights.

High-precision rules are applied first for particularly explicit account,
payment, refund, product, and delivery phrases.

This approach was chosen because it is inexpensive, reproducible, interpretable,
and appropriate for a relatively small labelled training set. A larger
fine-tuned language model was not necessary to establish a meaningful baseline.

## 7. Included two simple intent baselines

The evaluation compares the system against:

1. a trivial majority-class classifier; and
2. a simple keyword-rule classifier.

This makes it possible to measure whether the ML system provides value beyond
a naive approach.

The current held-out results are:

- trivial: accuracy 0.522, macro-F1 0.098
- keyword: accuracy 0.527, macro-F1 0.285
- system: accuracy 0.640, macro-F1 0.525

Macro-F1 is included because the golden set is strongly imbalanced.

## 8. Used TF-IDF retrieval for grounding

Historical customer/response pairs are indexed with TF-IDF and retrieved using
cosine similarity.

This was chosen instead of requiring an embedding API or vector database,
because the goal was to keep the take-home implementation lightweight and
fully reproducible offline.

## 9. Retrieved historical responses are used as reply evidence

The default reply strategy uses a high-similarity historical AmazonHelp
response as the basis for the draft.

The system records the retrieved evidence so that a reviewer can inspect the
source of the response.

This favors traceability and reproducibility over unrestricted generative
fluency.

## 10. Used a policy-based escalation mechanism

The current escalation policy always escalates:

- `account_access`
- `payment_billing`
- `refund_return`

It also escalates for serious legal/fraud/complaint language and for very low
retrieval similarity.

The policy is intentionally explicit rather than relying only on model
confidence.

## 11. Escalation ground truth is policy-derived

The escalation reference used in evaluation is derived from the labelled
intent and the designed escalation policy.

It is **not** an independent human judgement of whether an Amazon support agent
would escalate each case.

Therefore the escalation metrics measure consistency with the designed policy,
not real-world Amazon escalation correctness.

## 12. Kept replies available even when escalation is required

The pipeline produces a reply draft even when `escalate=True`.

The escalation flag determines whether the case should be automatically handled,
not whether a draft should exist.

A human agent can therefore review or edit the draft instead of starting from
an empty response.

## 13. Used an offline heuristic reply-quality judge by default

The reply-quality evaluation has a heuristic offline judge and an optional
Claude-based backend.

The headline evaluation was run without requiring an API key, using the
offline heuristic judge.

This makes the experiment reproducible without external API costs.

The limitation is explicitly documented: the heuristic judge is not equivalent
to a validated production LLM judge.

## 14. Performed a human-agreement study

A 40-example subset was manually rated using the same reply-quality dimensions.

The agreement results were:

- overall Pearson correlation: 0.42
- overall within-one-point agreement: 88%
- correctness Pearson correlation: 0.15
- tone Pearson correlation: 0.11
- completeness Pearson correlation: 0.38
- groundedness correlation: undefined because the automated groundedness
  scores had zero variance on the subset

This study was included to avoid presenting the automated judge as inherently
trustworthy without evidence.

## 15. Reported accuracy together with macro-F1

The golden set contains 106 `other` examples out of 203 total examples.

Because of this imbalance, accuracy alone could hide poor performance on
minority intents.

Macro-F1 is therefore reported alongside accuracy and is treated as the more
informative secondary classification metric.

## 16. Did not tune the final system on the golden labels

The golden set is used for evaluation rather than model training.

This avoids directly optimizing the classifier against the same examples used
to report its final performance.

## 17. Accepted limitations of the current prototype

The current system does not attempt to solve:

- multi-turn conversation state;
- authenticated account/order operations;
- production transaction handling;
- independently validated escalation labels;
- perfect sub-scenario retrieval;
- a fully validated LLM-as-judge evaluation.

These limitations are documented in `reports/report.md` rather than hidden
behind aggregate metrics.

## 18. Final reproducibility commands

The main evaluation is reproduced with:

```bash
python eval/run_eval.py
```

The human-agreement analysis is reproduced with:

```bash
python eval/judge_agreement.py
```

The automated tests are reproduced with:

```bash
python -m pytest -q
```

The end-to-end demo is run with:

```bash
python scripts/run_demo.py
```
