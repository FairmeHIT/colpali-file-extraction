#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DATASET_PATH = ROOT_DIR / "local_data" / "zh_corpus" / "benchmark_mineru" / "vidore_compat_test.jsonl"


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_local_dataset(path: Path, limit: int, include_unqueried: bool) -> Dataset:
    rows = _read_jsonl(path)
    if not include_unqueried:
        rows = [row for row in rows if row.get("query")]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows available after filtering `{path}`.")

    missing_images = [row["image"] for row in rows if not Path(row["image"]).exists()]
    if missing_images:
        preview = "\n".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} image paths are missing. First examples:\n{preview}")

    dataset = Dataset.from_list(rows).cast_column("image", Image())
    required = {"query", "image", "image_filename"}
    missing_columns = required.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrievers on the local MinerU zh_corpus benchmark.")
    parser.add_argument("--model-name", default="vidore/colpali")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-label", default="zh_corpus_mineru")
    parser.add_argument("--batch-query", type=int, default=1)
    parser.add_argument("--batch-doc", type=int, default=1)
    parser.add_argument("--batch-score", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N query rows for a smoke run. 0 means all.")
    parser.add_argument("--include-unqueried", action="store_true", help="Keep rows with query=None in the document pool.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "zh_corpus_mineru")
    parser.add_argument("--validate-only", action="store_true", help="Load and validate the local dataset without a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path if args.dataset_path.is_absolute() else ROOT_DIR / args.dataset_path

    print(f"Loading local dataset from `{dataset_path}`", flush=True)
    dataset = load_local_dataset(dataset_path, limit=args.limit, include_unqueried=args.include_unqueried)
    print(f"Loaded {len(dataset)} rows with columns: {dataset.column_names}", flush=True)
    if args.validate_only:
        first = dataset[0]
        image = first["image"]
        print(
            json.dumps(
                {
                    "status": "ok",
                    "rows": len(dataset),
                    "first_image_filename": first["image_filename"],
                    "first_query": first["query"],
                    "first_image_size": list(image.size),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    from vidore_benchmark.evaluation.evaluate import evaluate_dataset
    from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry

    import_retriever_for_model(args.model_name)
    retriever = load_vision_retriever_from_registry(args.model_name)()

    metrics = evaluate_dataset(
        retriever,
        dataset,  # type: ignore[arg-type]
        batch_query=args.batch_query,
        batch_doc=args.batch_doc,
        batch_score=args.batch_score,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model_name.replace("/", "_")
    limit_suffix = f"_limit_{args.limit}" if args.limit > 0 else ""
    output_path = args.output_dir / f"{model_slug}_{args.dataset_label}{limit_suffix}_metrics.json"
    payload = {args.dataset_label: metrics}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Metrics saved to `{output_path}`", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
