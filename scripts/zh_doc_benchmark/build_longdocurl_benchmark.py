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


DEFAULT_QUERY_KEYS = ["question", "query"]
DEFAULT_QUERY_ID_KEYS = ["question_id", "query_id", "id"]
DEFAULT_DOC_NO_KEYS = ["doc_no", "doc_id", "document_id"]
DEFAULT_EVIDENCE_PAGE_KEYS = ["evidence_pages", "evidence_page", "pages"]
DEFAULT_ANSWER_KEYS = ["answer", "answers"]
DEFAULT_IMAGE_LIST_KEYS = ["images", "image_paths", "image_files"]


def _candidate_page_paths(doc_no: str, page_no: int) -> list[str]:
    return [
        f"{doc_no}/{doc_no}_{page_no}.png",
        f"{doc_no}/{doc_no}_{page_no}.jpg",
        f"{doc_no}_{page_no}.png",
        f"{doc_no}_{page_no}.jpg",
        f"{doc_no}/{page_no}.png",
        f"{doc_no}/{page_no}.jpg",
    ]


def _build_sample(
    record: dict[str, Any],
    image_root: Path,
    image_index: dict[str, Path],
    source_path: Path,
    query_keys: list[str],
    query_id_keys: list[str],
    doc_no_keys: list[str],
    evidence_page_keys: list[str],
    answer_keys: list[str],
    image_list_keys: list[str],
) -> NormalizedSample | None:
    query = normalize_text(pick_first(record, query_keys))
    if not query:
        return None

    query_id = normalize_text(pick_first(record, query_id_keys)) or query
    doc_no = normalize_text(pick_first(record, doc_no_keys))
    answer = pick_first(record, answer_keys)
    evidence_pages = flatten_positive_pages(pick_first(record, evidence_page_keys))
    image_list = pick_first(record, image_list_keys)

    positive_image_paths: list[str] = []
    positive_doc_ids: list[str] = []

    if evidence_pages and doc_no:
        for page_no in evidence_pages:
            for candidate in _candidate_page_paths(doc_no, page_no):
                resolved = resolve_image_path(candidate, image_root, image_index)
                if resolved is not None:
                    doc_id = resolved.relative_to(image_root).as_posix()
                    positive_doc_ids.append(doc_id)
                    positive_image_paths.append(str(resolved))
                    break

    if not positive_doc_ids and isinstance(image_list, list):
        for candidate in image_list:
            resolved = resolve_image_path(candidate, image_root, image_index)
            if resolved is not None:
                doc_id = resolved.relative_to(image_root).as_posix()
                positive_doc_ids.append(doc_id)
                positive_image_paths.append(str(resolved))

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
        source_dataset="LongDocURL",
        source_record_path=str(source_path),
        source_record_id=normalize_text(pick_first(record, query_id_keys)) or None,
        evidence_pages=evidence_pages or None,
        doc_no=doc_no or None,
        task_tag=normalize_text(pick_first(record, ["task_tag"])) or None,
        question_type=normalize_text(pick_first(record, ["question_type"])) or None,
    )


def build_longdocurl_benchmark(args: argparse.Namespace) -> dict[str, Any]:
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
    doc_no_keys = split_keys(args.doc_no_keys, DEFAULT_DOC_NO_KEYS)
    evidence_page_keys = split_keys(args.evidence_page_keys, DEFAULT_EVIDENCE_PAGE_KEYS)
    answer_keys = split_keys(args.answer_keys, DEFAULT_ANSWER_KEYS)
    image_list_keys = split_keys(args.image_list_keys, DEFAULT_IMAGE_LIST_KEYS)

    samples: list[NormalizedSample] = []
    for record in records:
        sample = _build_sample(
            record,
            image_root=image_root,
            image_index=image_index,
            source_path=input_path,
            query_keys=query_keys,
            query_id_keys=query_id_keys,
            doc_no_keys=doc_no_keys,
            evidence_page_keys=evidence_page_keys,
            answer_keys=answer_keys,
            image_list_keys=image_list_keys,
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
            "evidence_pages": sample.evidence_pages,
            "doc_no": sample.doc_no,
            "task_tag": sample.task_tag,
            "question_type": sample.question_type,
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
            "evidence_pages": sample.evidence_pages,
            "doc_no": sample.doc_no,
            "task_tag": sample.task_tag,
            "question_type": sample.question_type,
            "source_dataset": sample.source_dataset,
            "source_record_path": sample.source_record_path,
            "source_record_id": sample.source_record_id,
        }
        for sample in samples
    ]
    write_jsonl(benchmark_root / "vidore_compat_test.jsonl", vidore_rows)

    write_jsonl(processed_root / "normalized_records.jsonl", [asdict(sample) for sample in samples])
    report = {
        "dataset": "LongDocURL",
        "inputs": {
            "input": str(input_path.resolve()),
            "image_root": str(image_root.resolve()),
        },
        "counts": {
            "records_loaded": len(records),
            "samples_kept": len(samples),
            "corpus_images": len(corpus_rows),
            "multi_positive_queries": sum(1 for sample in samples if len(sample.positive_doc_ids) > 1),
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
    parser = argparse.ArgumentParser(description="Build an isolated LongDocURL benchmark draft.")
    parser.add_argument("--input", required=True, help="Path to LongDocURL.jsonl or an extracted metadata directory.")
    parser.add_argument("--image-root", required=True, help="Root directory containing extracted page images.")
    parser.add_argument(
        "--out-root",
        default="local_data/zh_doc_benchmark/longdocurl",
        help="Output root for the isolated benchmark artifacts.",
    )
    parser.add_argument("--query-keys", default=",".join(DEFAULT_QUERY_KEYS))
    parser.add_argument("--query-id-keys", default=",".join(DEFAULT_QUERY_ID_KEYS))
    parser.add_argument("--doc-no-keys", default=",".join(DEFAULT_DOC_NO_KEYS))
    parser.add_argument("--evidence-page-keys", default=",".join(DEFAULT_EVIDENCE_PAGE_KEYS))
    parser.add_argument("--answer-keys", default=",".join(DEFAULT_ANSWER_KEYS))
    parser.add_argument("--image-list-keys", default=",".join(DEFAULT_IMAGE_LIST_KEYS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-images", action="store_true", default=False)
    parser.add_argument("--allow-empty-queries", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_longdocurl_benchmark(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
