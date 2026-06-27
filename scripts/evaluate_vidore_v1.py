#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset

from vidore_benchmark.evaluation.evaluate import evaluate_dataset
from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry


VIDORE_V1_DATASETS = [
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
    "vidore/syntheticDocQA_artificial_intelligence_test",
    "vidore/syntheticDocQA_energy_test",
    "vidore/syntheticDocQA_government_reports_test",
    "vidore/syntheticDocQA_healthcare_industry_test",
]


def import_retriever_for_model(model_name: str) -> None:
    if model_name == "vidore/colpali":
        import vidore_benchmark.retrievers.colpali_retriever  # noqa: F401
    elif model_name == "google/siglip-so400m-patch14-384":
        import vidore_benchmark.retrievers.siglip_retriever  # noqa: F401
    elif model_name == "jinaai/jina-clip-v1":
        import vidore_benchmark.retrievers.jina_clip_retriever  # noqa: F401
    elif model_name == "nomic-ai/nomic-embed-vision-v1.5":
        import vidore_benchmark.retrievers.nomic_retriever  # noqa: F401
    elif model_name == "bm25":
        import vidore_benchmark.retrievers.bm25_retriever  # noqa: F401
    elif model_name == "BAAI/bge-m3":
        import vidore_benchmark.retrievers.bge_m3_retriever  # noqa: F401
    elif model_name == "BAAI/bge-m3-colbert":
        import vidore_benchmark.retrievers.bge_m3_colbert_retriever  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retriever on ViDoRe V1 without using HF collection API.")
    parser.add_argument("--model-name", default="vidore/colpali")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-query", type=int, default=4)
    parser.add_argument("--batch-doc", type=int, default=4)
    parser.add_argument("--batch-score", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-name", action="append", dest="datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or VIDORE_V1_DATASETS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model_name.replace("/", "_")
    model_dir = args.output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    import_retriever_for_model(args.model_name)
    retriever = load_vision_retriever_from_registry(args.model_name)()

    all_metrics: dict[str, dict[str, float]] = {}
    all_path = args.output_dir / f"{model_slug}_all_metrics.json"

    for dataset_name in datasets:
        print(f"\n---------------------------\nEvaluating {dataset_name}", flush=True)
        dataset = load_dataset(dataset_name, split=args.split)
        metrics = evaluate_dataset(
            retriever,
            dataset,  # type: ignore[arg-type]
            batch_query=args.batch_query,
            batch_doc=args.batch_doc,
            batch_score=args.batch_score,
        )

        all_metrics[dataset_name] = metrics

        dataset_path = model_dir / f"{dataset_name.replace('/', '_')}_metrics.json"
        dataset_path.write_text(json.dumps({dataset_name: metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
        all_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Metrics saved to `{dataset_path}`", flush=True)
        print(f"NDCG@5 for {args.model_name} on {dataset_name}: {metrics['ndcg_at_5']}", flush=True)

    print(f"\nConcatenated metrics saved to `{all_path}`")
    print("Done.")


if __name__ == "__main__":
    main()
