# zh_corpus MinerU Benchmark Experiment Notes

Date: 2026-06-16

## Context

We built a local Chinese visual document retrieval benchmark from three raw datasets under `local_data/zh_corpus/raw/`:

| Dataset | Images | Notes |
|---|---:|---|
| `COLD_CELL_600` | 600 | receipts, reports, small business documents |
| `EPHOIE_SCUT_311` | 311 | Chinese extraction seed |
| `XFUND-zh-val` | 50 | Chinese form pages |

MinerU flash OCR/layout markdown exists for 611 pages:

| Dataset | MinerU Markdown Pages |
|---|---:|
| `COLD_CELL_600` | 561 |
| `XFUND-zh-val` | 49 |
| `EPHOIE_SCUT_311` | 1 |

The MinerU-content benchmark was generated with:

```bash
source env.sh
python scripts/build_zh_mineru_benchmark.py --max-evidence-chars 60
```

Outputs:

```text
local_data/zh_corpus/benchmark_mineru/
  corpus.jsonl
  queries.jsonl
  qrels.json
  selected_queries.jsonl
  query_candidates.jsonl
  mineru_page_texts.jsonl
  vidore_compat_test.jsonl
```

Final benchmark counts:

| Item | Count |
|---|---:|
| MinerU pages with text | 611 |
| Final selected queries | 506 |
| Pages without selected query | 105 |

Selected query distribution:

| Dataset | Selected Queries |
|---|---:|
| `COLD_CELL_600` | 468 |
| `XFUND-zh-val` | 37 |
| `EPHOIE_SCUT_311` | 1 |

Query evidence types:

| Type | Count |
|---|---:|
| identifier | 159 |
| title | 137 |
| date | 119 |
| phone | 65 |
| text | 14 |
| money | 12 |

## ColPali Result

Command:

```bash
source env.sh
python scripts/evaluate_zh_mineru_benchmark.py \
  --model-name vidore/colpali \
  --batch-query 1 \
  --batch-doc 1 \
  --batch-score 1
```

Output:

```text
outputs/zh_corpus_mineru/vidore_colpali_zh_corpus_mineru_metrics.json
```

Key metrics:

| Metric | Value |
|---|---:|
| NDCG@1 | 0.15613 |
| NDCG@5 | 0.21837 |
| NDCG@10 | 0.24579 |
| Recall@1 | 0.15613 |
| Recall@5 | 0.27668 |
| Recall@20 | 0.46640 |
| Recall@100 | 0.77470 |
| MRR@10 | 0.20367 |

Interpretation:

ColPali shows non-trivial coarse retrieval ability because Recall@100 reaches 0.77470, but top-rank quality is weak. For this benchmark, it often finds the correct page somewhere in the larger candidate set but does not rank it near the top.

## BM25 Result

Command:

```bash
source env.sh
python scripts/evaluate_zh_mineru_hybrid.py --mode bm25
```

Output:

```text
outputs/zh_corpus_mineru/bm25_zh_corpus_mineru_bm25_metrics.json
```

Key metrics:

| Metric | Value |
|---|---:|
| NDCG@1 | 0.70751 |
| NDCG@5 | 0.77233 |
| NDCG@10 | 0.78404 |
| Recall@1 | 0.70751 |
| Recall@5 | 0.82213 |
| Recall@20 | 0.89526 |
| Recall@100 | 0.94664 |
| MRR@10 | 0.76038 |

Interpretation:

BM25 is much stronger than ColPali on this benchmark because the benchmark queries are mostly generated from MinerU OCR text. BM25 does not perform OCR itself; it uses the already extracted `text_description` field from MinerU and performs lexical matching between query text and page text.

## Main Finding

The current `benchmark_mineru` construction is biased toward OCR lexical retrieval.

Most queries are exact-field lookup templates such as:

```text
查找包含编号“中心编号：WT2019B01A04477”的中文文档页面。
查找包含联系电话“020-85511833”的中文文档页面。
查找包含日期“2016-12-20”的中文文档页面。
```

These queries have high token overlap with MinerU OCR text. As a result, BM25 has a direct advantage. This benchmark is useful for evaluating OCR + lexical retrieval, but it is not yet a high-quality primary benchmark for ColPali-style visual document retrieval.

## What This Means

The weak ColPali score should not be interpreted as proof that ColPali is ineffective for Chinese visual document retrieval. It primarily shows that the current query construction is not well matched to ColPali's strengths.

Current benchmark mainly tests:

```text
exact OCR field matching
```

It does not sufficiently test:

```text
visual layout understanding
document type recognition
semantic page retrieval
table/receipt/report structure understanding
```

## Recommended Dataset Redesign

Keep three query splits:

### 1. exact-field split

Use the current field-style queries. This split is BM25-friendly and should be treated as an OCR lexical retrieval diagnostic.

Examples:

```text
查找包含编号“102310054111207”的中文文档页面。
查找包含日期“2016-12-20”的中文文档页面。
```

### 2. semantic-layout split

Rewrite queries into natural descriptions without directly copying unique OCR strings.

Examples:

```text
查找一张银联商务消费签购单页面。
查找关于跑道材料检测结论的报告首页。
查找列出样品名称、检验类别和判定依据的检测报告页面。
查找一张餐厅等位优惠说明小票。
```

This split should be the main split for evaluating ColPali.

### 3. hybrid-realistic split

Combine document type or scenario with a small amount of non-unique field evidence.

Examples:

```text
查找大润发马鞍山店的一张银联消费小票。
查找北京大学基建工程部委托检验的建筑材料检测报告。
查找金额约 83 元的消费凭证。
```

This split is closer to realistic retrieval and should benefit from ColPali + BM25 hybrid.

## Hybrid Direction

Hybrid evaluation was implemented locally in:

```text
scripts/evaluate_zh_mineru_hybrid.py
```

It supports:

```bash
python scripts/evaluate_zh_mineru_hybrid.py --mode bm25
python scripts/evaluate_zh_mineru_hybrid.py --mode rrf --model-name vidore/colpali
python scripts/evaluate_zh_mineru_hybrid.py --mode linear --model-name vidore/colpali --alpha 0.5
```

No official ColPali or ViDoRe source files are modified. Hybrid scoring is isolated in local scripts.

Fusion methods:

| Mode | Description |
|---|---|
| `bm25` | BM25 over MinerU `text_description` only |
| `rrf` | Reciprocal Rank Fusion of ColPali rank and BM25 rank |
| `linear` | Min-max normalized score interpolation |

## Next Step

Do not use the current `benchmark_mineru` as the final primary Chinese ColPali benchmark. Use it as a diagnostic baseline first.

Next implementation should generate a new query set with explicit split labels:

```text
exact_field
semantic_layout
hybrid_realistic
```

The final report should compare:

| Method | exact_field | semantic_layout | hybrid_realistic |
|---|---:|---:|---:|
| ColPali | expected weak | target split | useful |
| BM25 | expected strong | likely weaker | useful |
| ColPali + BM25 RRF | expected strong | likely robust | target method |

