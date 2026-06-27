#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from zh_doc_benchmark.common import (
    NormalizedSample,
    build_corpus_rows,
    ensure_dir,
    flatten_positive_pages,
    index_images,
    load_records,
    normalize_text,
    pick_first,
    resolve_image_path,
    split_keys,
    write_jsonl,
)


DEFAULT_QUERY_KEYS = ["query", "question", "input"]
DEFAULT_QUERY_ID_KEYS = ["question_id", "query_id", "qid", "id"]
DEFAULT_IMAGE_KEYS = ["image", "image_path", "image_filename", "doc_image", "page_image"]
DEFAULT_DOC_ID_KEYS = ["doc_id", "docId", "document_id", "doc_no", "docno"]
DEFAULT_PAGE_KEYS = ["page", "page_no", "page_id"]
DEFAULT_ANSWER_KEYS = ["answer", "answers", "gold_answer"]


def _build_sample(
    record: dict[str, Any],
    image_root: Path,
    image_index: dict[str, Path],
    source_path: Path,
    query_keys: list[str],
    query_id_keys: list[str],
    image_keys: list[str],
    doc_id_keys: list[str],
    page_keys: list[str],
    answer_keys: list[str],
) -> NormalizedSample | None:
    query = normalize_text(pick_first(record, query_keys))
    if not query:
        return None

    query_id = normalize_text(pick_first(record, query_id_keys)) or query
    answer = pick_first(record, answer_keys)

    image_value = pick_first(record, image_keys)
    doc_id = normalize_text(pick_first(record, doc_id_keys))
    page_value = pick_first(record, page_keys)

    positive_doc_ids: list[str] = []
    positive_image_paths: list[str] = []

    resolved = resolve_image_path(image_value, image_root, image_index) if image_value is not None else None
    if resolved is not None:
        positive_doc_ids = [resolved.relative_to(image_root).as_posix()]
        positive_image_paths = [str(resolved)]
    elif doc_id and page_value is not None:
        page_str = normalize_text(page_value)
        candidates = [
            f"{doc_id}/{page_str}.png",
            f"{doc_id}/{page_str}.jpg",
            f"{doc_id}_{page_str}.png",
            f"{doc_id}_{page_str}.jpg",
            f"{doc_id}/{doc_id}_{page_str}.png",
            f"{doc_id}/{doc_id}_{page_str}.jpg",
        ]
        for candidate in candidates:
            resolved = resolve_image_path(candidate, image_root, image_index)
            if resolved is not None:
                positive_doc_ids = [resolved.relative_to(image_root).as_posix()]
                positive_image_paths = [str(resolved)]
                break
    elif doc_id:
        candidates = [f"{doc_id}.png", f"{doc_id}.jpg", doc_id]
        for candidate in candidates:
            resolved = resolve_image_path(candidate, image_root, image_index)
            if resolved is not None:
                positive_doc_ids = [resolved.relative_to(image_root).as_posix()]
                positive_image_paths = [str(resolved)]
                break

    if not positive_doc_ids:
        return None

    anchor_doc_id = positive_doc_ids[0]
    anchor_image_path = positive_image_paths[0]

    return NormalizedSample(
        query_id=query_id,
        query=query,
        anchor_doc_id=anchor_doc_id,
        anchor_image_path=anchor_image_path,
        anchor_image_filename=anchor_doc_id,
        positive_doc_ids=positive_doc_ids,
        positive_image_paths=positive_image_paths,
        answer=answer,
        source_dataset="DuReader-vis",
        source_record_path=str(source_path),
        source_record_id=normalize_text(pick_first(record, query_id_keys)) or None,
        evidence_pages=flatten_positive_pages(pick_first(record, ["evidence_pages", "evidence_page", "pages"])),
        doc_no=doc_id or None,
        question_type=normalize_text(pick_first(record, ["question_type", "qtype"])) or None,
    )


def build_dureader_vis_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    image_root = Path(args.image_root)
    out_root = Path(args.out_root)
    processed_root = ensure_dir(out_root / "processed")
    benchmark_root = ensure_dir(out_root / "benchmark")
    reports_root = ensure_dir(out_root / "reports")

    image_index = index_images(image_root)
    records = load_records(input_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    query_keys = split_keys(args.query_keys, DEFAULT_QUERY_KEYS)
    query_id_keys = split_keys(args.query_id_keys, DEFAULT_QUERY_ID_KEYS)
    image_keys = split_keys(args.image_keys, DEFAULT_IMAGE_KEYS)
    doc_id_keys = split_keys(args.doc_id_keys, DEFAULT_DOC_ID_KEYS)
    page_keys = split_keys(args.page_keys, DEFAULT_PAGE_KEYS)
    answer_keys = split_keys(args.answer_keys, DEFAULT_ANSWER_KEYS)

    samples: list[NormalizedSample] = []
    for record in records:
        sample = _build_sample(
            record,
            image_root=image_root,
            image_index=image_index,
            source_path=input_path,
            query_keys=query_keys,
            query_id_keys=query_id_keys,
            image_keys=image_keys,
            doc_id_keys=doc_id_keys,
            page_keys=page_keys,
            answer_keys=answer_keys,
        )
        if sample is not None:
            samples.append(sample)

    if args.require_images and not image_index:
        raise FileNotFoundError(f"No images found under {image_root}")
    if not args.allow_empty_queries and not samples:
        raise ValueError("No query rows were normalized from the provided input.")

    corpus_rows = build_corpus_rows(image_root)
    write_jsonl(benchmark_root / "corpus.jsonl", corpus_rows)

    queries_rows = [
        {
            "query_id": sample.query_id,
            "query": sample.query,
            "answer": sample.answer,
            "anchor_doc_id": sample.anchor_doc_id,
            "positive_doc_ids": sample.positive_doc_ids,
            "source_dataset": sample.source_dataset,
            "source_record_path": sample.source_record_path,
            "source_record_id": sample.source_record_id,
        }
        for sample in samples
    ]
    write_jsonl(benchmark_root / "queries.jsonl", queries_rows)

    qrels = {sample.query_id: {doc_id: 1 for doc_id in sample.positive_doc_ids} for sample in samples}
    (benchmark_root / "qrels.json").write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")

    vidore_rows = [
        {
            "query_id": sample.query_id,
            "query": sample.query,
            "image": sample.anchor_image_path,
            "image_filename": sample.anchor_image_filename,
            "doc_id": sample.anchor_doc_id,
            "answer": sample.answer,
            "positive_doc_ids": sample.positive_doc_ids,
            "positive_image_paths": sample.positive_image_paths,
            "source_dataset": sample.source_dataset,
            "source_record_path": sample.source_record_path,
            "source_record_id": sample.source_record_id,
        }
        for sample in samples
    ]
    write_jsonl(benchmark_root / "vidore_compat_test.jsonl", vidore_rows)

    write_jsonl(processed_root / "normalized_records.jsonl", [asdict(sample) for sample in samples])
    report = {
        "dataset": "DuReader-vis",
        "inputs": {
            "input": str(input_path.resolve()),
            "image_root": str(image_root.resolve()),
        },
        "counts": {
            "records_loaded": len(records),
            "samples_kept": len(samples),
            "corpus_images": len(corpus_rows),
        },
        "outputs": {
            "processed_records": str((processed_root / "normalized_records.jsonl").resolve()),
            "corpus": str((benchmark_root / "corpus.jsonl").resolve()),
            "queries": str((benchmark_root / "queries.jsonl").resolve()),
            "qrels": str((benchmark_root / "qrels.json").resolve()),
            "vidore_compat_test": str((benchmark_root / "vidore_compat_test.jsonl").resolve()),
        },
    }
    (reports_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated DuReader-vis benchmark draft.")
    parser.add_argument("--input", required=True, help="Path to the extracted DuReader-vis metadata file or directory.")
    parser.add_argument("--image-root", required=True, help="Root directory containing the document images.")
    parser.add_argument(
        "--out-root",
        default="local_data/zh_doc_benchmark/dureader_vis",
        help="Output root for the isolated benchmark artifacts.",
    )
    parser.add_argument("--query-keys", default=",".join(DEFAULT_QUERY_KEYS))
    parser.add_argument("--query-id-keys", default=",".join(DEFAULT_QUERY_ID_KEYS))
    parser.add_argument("--image-keys", default=",".join(DEFAULT_IMAGE_KEYS))
    parser.add_argument("--doc-id-keys", default=",".join(DEFAULT_DOC_ID_KEYS))
    parser.add_argument("--page-keys", default=",".join(DEFAULT_PAGE_KEYS))
    parser.add_argument("--answer-keys", default=",".join(DEFAULT_ANSWER_KEYS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-images", action="store_true", default=False)
    parser.add_argument("--allow-empty-queries", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dureader_vis_benchmark(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
