# Chunking strategy comparison

Generated 2026-10-28 14:47. Bold = best value in the column.

## Setup

- Test cases: 56 (same questions for every strategy)
- Chunk size / overlap: 500 / 100 characters
- Embeddings: local, generator: llama3.2, judge: openai/gpt-oss-120b
- Mode: full pipeline with LLM judge

| Strategy | Chunks | Avg chars | Min | Max | Duplicates skipped |
|---|---|---|---|---|---|
| fixed | 24 | 306.8 | 220 | 387 | 0 |
| recursive | 24 | 306.8 | 220 | 387 | 0 |
| semantic | 25 | 294.4 | 75 | 387 | 0 |

## Overall

| Strategy | Recall | MRR | Ctx precision | Correctness | Faithfulness | Citation acc. | False refusal | Hallucinated | Score | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 1.00 | 0.99 | 0.27 | 0.79 | 0.98 | **0.95** | 0.11 | 0.00 | 0.93 | 17.9 |
| recursive | 1.00 | 0.99 | 0.27 | 0.79 | 0.98 | **0.95** | 0.11 | 0.00 | 0.93 | **1.0** |
| semantic | 1.00 | 0.99 | 0.27 | **0.81** | **0.98** | 0.95 | 0.11 | 0.00 | **0.93** | 10.2 |

## By question type

### ambiguous (n=8)

| Strategy | Recall | Correctness | Faithfulness | Citation acc. |
|---|---|---|---|---|
| fixed | 1.00 | 0.44 | 0.97 | 1.00 |
| recursive | 1.00 | 0.44 | 0.97 | 1.00 |
| semantic | 1.00 | 0.44 | 0.97 | 1.00 |

### lookup (n=28)

| Strategy | Recall | Correctness | Faithfulness | Citation acc. |
|---|---|---|---|---|
| fixed | 1.00 | 0.89 | 1.00 | **0.96** |
| recursive | 1.00 | 0.89 | 1.00 | **0.96** |
| semantic | 1.00 | 0.89 | 1.00 | 0.95 |

### multi_hop (n=10)

| Strategy | Recall | Correctness | Faithfulness | Citation acc. |
|---|---|---|---|---|
| fixed | 1.00 | 0.60 | 0.90 | 0.88 |
| recursive | 1.00 | 0.60 | 0.90 | 0.88 |
| semantic | 1.00 | **0.70** | **0.93** | 0.88 |

### no_answer (n=10)

| Strategy | Recall | Correctness | Faithfulness | Citation acc. |
|---|---|---|---|---|
| fixed | - | 1.00 | - | - |
| recursive | - | 1.00 | - | - |
| semantic | - | 1.00 | - | - |

## Winners

- **Retrieval recall:** tie between fixed, recursive, semantic (1.00)
- **Context precision:** tie between fixed, recursive, semantic (0.27)
- **Answer correctness:** semantic (0.81)
- **Faithfulness:** semantic (0.98)
- **Citation accuracy:** tie between fixed, recursive (0.95)
- **Overall score:** semantic (0.93)

## How to read this

- Recall / context precision measure retrieval; correctness, faithfulness and citation accuracy are judged by an LLM and are only as reliable as the judge model.
- With a small test set, differences under about 0.05 are within noise. Add more documents and questions before drawing strong conclusions.
- False refusal = answerable questions the system refused. Hallucinated = unanswerable questions that received an answer instead of a refusal. Lower is better for both.
