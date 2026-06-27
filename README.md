# ColPali 图文交错长文档高效解析技术攻关

基于 MinerU 多模态解析引擎与 BM25 检索验证框架的文档解析方案，支持将 DOC/PDF/Excel 等常见文档解析为适合大模型处理的 Markdown 格式。

## 核心指标

| 方案 | 基准 | nDCG@5 |
|------|------|--------|
| BM25 文本检索 | 中文 MinerU 基准 (506 查询) | 77.23% |
| ColPali+BM25 线性融合 | 中文 MinerU 基准 (506 查询) | 76.63% |
| ColQwen2 视觉检索 | 中文证照基准 (13 类) | 89.59% |

## 项目结构

```
├── scripts/                     # 核心评测与基准构建脚本
│   ├── enrich_zh_corpus_with_mineru.py    # 文档解析管道 (MinerU API)
│   ├── build_zh_corpus_manifest.py        # 语料清单构建
│   ├── build_zh_mineru_benchmark.py       # 评测基准自动构建
│   ├── evaluate_zh_mineru_benchmark.py    # ViDoRe 标准评测
│   ├── evaluate_zh_mineru_hybrid.py       # BM25 + ColPali 融合评测
│   ├── evaluate_zh_cert_benchmark.py      # 证照检索评测
│   ├── evaluate_vidore_v1.py              # ViDoRe V1 国际基准评测
│   └── zh_doc_benchmark/                  # 外部数据集适配
├── custom_benchmarks/           # 自定义检索器
│   └── colqwen2_retriever.py             # ColQwen2 检索器封装
├── local_data/                  # 评测基准
│   └── zh_corpus/benchmark_mineru/       # 中文 MinerU 基准 (506 查询/611 页)
├── outputs/                     # 评测结果
├── reports/                     # 验收报告
├── env.sh / env_colqwen2.sh    # 环境配置
└── run_*.sh                     # 一键评测脚本
```

## 环境配置

```bash
# 主评测环境 (ColPali)
python -m venv .venv-vidore
source .venv-vidore/bin/activate
uv pip install -e vidore-benchmark/

# ColQwen2 环境 (需要新版本 transformers)
bash setup_colqwen2_env.sh
```

## 运行评测

```bash
source env.sh

# BM25 文本检索 (主验收方案)
python scripts/evaluate_zh_mineru_hybrid.py --mode bm25

# ColPali+BM25 线性融合
python scripts/evaluate_zh_mineru_hybrid.py --mode linear --model-name vidore/colpali --alpha 0.5

# 证照基准 (ColQwen2)
source env_colqwen2.sh
bash run_zh_cert_colqwen2_smoke.sh
```

## 依赖

- Python 3.12, PyTorch 2.2+, Transformers 4.41+
- colpali-engine 0.1.1 (主环境) / 0.3.7 (ColQwen2 环境)
- vidore-benchmark 3.3.0, MTEB 1.12+
- mineru-open-sdk


