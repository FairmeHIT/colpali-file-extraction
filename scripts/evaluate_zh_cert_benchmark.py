#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, Image
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DATA_ROOT = ROOT_DIR / "local_data" / "zh_cert_benchmark"
OUTPUT_ROOT = DATA_ROOT / "benchmark_runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CLASS_QUERY_NAMES = {
    "businessLicense": "营业执照",
    "ID": "身份证",
    "academicCertificate": "学历证书",
    "degreeCertificate": "学位证书",
    "legalLicense": "事业单位法人证",
    "ISO9000": "质量管理体系认证证书ISO9001",
    "ISO14000": "环境管理体系认证证书ISO14001",
    "ISO45001": "ISO45001职业健康安全管理体系认证证书",
    "SA8000": "社会责任管理体系认证证书",
    "CESSCN_design_inte": "通信网络安全服务能力评定证书（安全设计与集成）",
    "CESSCN_emergency_resp": "通信网络安全服务能力评定证书（应急响应）",
    "CESSCN_risk_eval": "通信网络安全服务能力评定证书（风险评估）",
    "CESSCN_safety_train": "通信网络安全服务能力评定证书（安全培训）",
}


def import_retriever_for_model(model_name: str) -> None:
    if model_name == "vidore/colpali":
        import vidore_benchmark.retrievers.colpali_retriever  # noqa: F401
    elif model_name == "google/siglip-so400m-patch14-384":
        import vidore_benchmark.retrievers.siglip_retriever  # noqa: F401
    elif model_name == "jinaai/jina-clip-v1":
        import vidore_benchmark.retrievers.jina_clip_retriever  # noqa: F401
    elif model_name == "nomic-ai/nomic-embed-vision-v1.5":
        import vidore_benchmark.retrievers.nomic_retriever  # noqa: F401


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_classes(data_root: Path) -> list[str]:
    classes = []
    for path in sorted(data_root.iterdir(), key=lambda p: p.name):
        if path.is_dir() and path.name != "others" and list_images(path):
            classes.append(path.name)
    return classes


def parse_classes(raw: str | None, data_root: Path) -> list[str]:
    if not raw:
        return discover_classes(data_root)
    classes = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [cls for cls in classes if not (data_root / cls).is_dir()]
    if missing:
        raise FileNotFoundError(f"Unknown class directories: {missing}")
    return classes


def make_query(class_name: str) -> str:
    display = CLASS_QUERY_NAMES.get(class_name, class_name)
    return f"查找一张{display}图片。"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def build_run_dataset(
    data_root: Path,
    output_root: Path,
    n_per_class: int,
    n_others: int,
    classes: list[str],
    seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    selected: dict[str, list[str]] = {}

    for class_name in classes:
        images = list_images(data_root / class_name)
        if not images:
            continue
        if n_per_class > len(images):
            raise ValueError(f"Class `{class_name}` has only {len(images)} images, cannot sample {n_per_class}.")
        sampled = rng.sample(images, n_per_class)
        selected[class_name] = [str(path.resolve()) for path in sampled]
        query = make_query(class_name)
        for path in sampled:
            doc_id = f"{class_name}/{path.name}"
            rows.append(
                {
                    "query": query,
                    "image": str(path.resolve()),
                    "image_filename": doc_id,
                    "doc_id": doc_id,
                    "class_name": class_name,
                    "target_class": class_name,
                    "is_positive": True,
                }
            )

    others = list_images(data_root / "others")
    if n_others < 0:
        n_others = len(others)
    if n_others > len(others):
        raise ValueError(f"`others` has only {len(others)} images, cannot sample {n_others}.")
    sampled_others = rng.sample(others, n_others)
    selected["others"] = [str(path.resolve()) for path in sampled_others]
    for path in sampled_others:
        doc_id = f"others/{path.name}"
        rows.append(
            {
                "query": None,
                "image": str(path.resolve()),
                "image_filename": doc_id,
                "doc_id": doc_id,
                "class_name": "others",
                "target_class": None,
                "is_positive": False,
            }
        )

    rng.shuffle(rows)

    run_name = f"seed_{seed}_n_{n_per_class}_others_{n_others}"
    run_root = output_root / run_name
    dataset_path = run_root / "vidore_compat_test.jsonl"
    manifest_path = run_root / "manifest.json"
    write_jsonl(dataset_path, rows)

    report = {
        "seed": seed,
        "n_per_class": n_per_class,
        "n_others": n_others,
        "classes": classes,
        "counts": {
            "rows": len(rows),
            "queries": len(classes),
            "positive_images": sum(1 for row in rows if row["is_positive"]),
            "others": len(sampled_others),
        },
        "selected": selected,
        "dataset_path": str(dataset_path.resolve()),
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_path, manifest_path, report


def load_dataset_for_eval(dataset_path: Path) -> Dataset:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in `{dataset_path}`.")
    missing = [row["image"] for row in rows if not Path(row["image"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image paths missing. First: {missing[0]}")
    return Dataset.from_list(rows).cast_column("image", Image())


def build_relevant_docs(ds: Dataset) -> dict[str, dict[str, int]]:
    query_to_class = {row["query"]: row["target_class"] for row in ds if row.get("query")}
    relevant_docs: dict[str, dict[str, int]] = {}
    for query, class_name in query_to_class.items():
        positives = {
            row["image_filename"]: 1
            for row in ds
            if row.get("class_name") == class_name and row.get("is_positive")
        }
        relevant_docs[query] = positives
    return relevant_docs


def unique_queries(ds: Dataset) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for row in ds:
        query = row.get("query")
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
    return queries


def scores_to_results(scores: Any, queries: list[str], doc_ids: list[str]) -> dict[str, dict[str, float]]:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    results: dict[str, dict[str, float]] = {}
    for qidx, query in enumerate(queries):
        results[query] = {doc_id: float(scores[qidx][didx]) for didx, doc_id in enumerate(doc_ids)}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random certificate-vs-others ColPali evaluation.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--classes", default=None, help="Comma-separated class directories. Default: all except others.")
    parser.add_argument("--n-per-class", type=int, default=3)
    parser.add_argument("--n-others", type=int, default=30, help="Use -1 for all images in others.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="vidore/colpali")
    parser.add_argument(
        "--backend",
        choices=["registry", "colqwen2"],
        default="registry",
        help="Use upstream ViDoRe registry retrievers or the isolated local ColQwen2 retriever.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-query", type=int, default=1)
    parser.add_argument("--batch-doc", type=int, default=1)
    parser.add_argument("--batch-score", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run-build", action="store_true", help="Build dataset files but do not load a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root if args.data_root.is_absolute() else ROOT_DIR / args.data_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT_DIR / args.output_root
    classes = parse_classes(args.classes, data_root)
    dataset_path, manifest_path, report = build_run_dataset(
        data_root=data_root,
        output_root=output_root,
        n_per_class=args.n_per_class,
        n_others=args.n_others,
        classes=classes,
        seed=args.seed,
    )
    print(json.dumps({"dataset_path": str(dataset_path), "manifest_path": str(manifest_path), **report["counts"]}, ensure_ascii=False, indent=2))

    dataset = load_dataset_for_eval(dataset_path)
    relevant_docs = build_relevant_docs(dataset)
    if args.validate_only or args.dry_run_build:
        first = dataset[0]
        print(
            json.dumps(
                {
                    "status": "ok",
                    "rows": len(dataset),
                    "queries": len(relevant_docs),
                    "class_counts": dict(Counter(dataset["class_name"])),
                    "first_image_filename": first["image_filename"],
                    "first_image_size": list(first["image"].size),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.backend == "colqwen2":
        from custom_benchmarks.colqwen2_retriever import LocalColQwen2Retriever

        retriever = LocalColQwen2Retriever(
            pretrained_model_name_or_path=args.model_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    else:
        from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry

        import_retriever_for_model(args.model_name)
        retriever = load_vision_retriever_from_registry(args.model_name)()
    queries = unique_queries(dataset)
    documents = list(dataset["image"]) if retriever.use_visual_embedding else list(dataset["text_description"])
    emb_queries = retriever.forward_queries(queries, batch_size=args.batch_query)
    emb_documents = retriever.forward_documents(documents, batch_size=args.batch_doc)
    scores = retriever.get_scores(emb_queries, emb_documents, batch_size=args.batch_score)
    results = scores_to_results(scores, queries, list(dataset["image_filename"]))
    metrics = retriever.compute_metrics(relevant_docs, results)

    model_slug = f"{args.backend}_{args.model_name.replace('/', '_')}"
    metrics_path = manifest_path.parent / f"{model_slug}_metrics.json"
    payload = {
        "zh_cert_benchmark": {
            "backend": args.backend,
            "model_name": args.model_name,
            "dataset_path": str(dataset_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "relevance": "all sampled images from the requested class are positives for the class query",
            "metrics": metrics,
        }
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metrics saved to `{metrics_path}`")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
