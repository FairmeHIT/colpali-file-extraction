#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


CORPUS_ROOT = ROOT_DIR / "local_data" / "zh_corpus"
PROCESSED_ROOT = CORPUS_ROOT / "processed"
BENCHMARK_ROOT = CORPUS_ROOT / "benchmark"
REPORTS_ROOT = CORPUS_ROOT / "reports"

GENERIC_FIELD_NAMES = {
    "no",
    "no.",
    "编号",
    "单号",
    "页码",
    "备注",
    "注",
    "日期",
    "时间",
    "合计",
    "合计.数量",
    "合计.金额",
    "标题",
}


@dataclass
class QueryCandidate:
    query_id: str
    query: str
    page_id: str
    dataset: str
    field_name: str
    field_value: str
    query_value: str
    value_length: int
    score: float


@dataclass
class BenchmarkRow:
    query: str | None
    image: str
    image_filename: str
    page_id: str
    dataset: str
    source_image_path: str
    source_json_path: str
    field_name: str | None = None
    field_value: str | None = None
    query_id: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _shorten_value(value: str, max_chars: int) -> str:
    value = _normalize_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("，,。；;、 ") + "..."


def _has_signal(value: str, min_value_length: int) -> bool:
    value = _normalize_text(value)
    if len(value) < min_value_length:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", value):
        return False
    return True


def _make_query(field_name: str, query_value: str) -> str:
    return f"查找字段“{field_name}”内容为“{query_value}”的中文文档页面。"


def _candidate_score(field_name: str, field_value: str, value_page_freq: int) -> float:
    value_len = len(field_value)
    score = 0.0

    if 14 <= value_len <= 80:
        score += 4.0
    elif 10 <= value_len < 14:
        score += 2.0
    elif 80 < value_len <= 180:
        score += 2.5
    else:
        score += 1.0

    if re.search(r"[\u4e00-\u9fff]", field_value):
        score += 2.0
    if re.search(r"[A-Za-z0-9]", field_value):
        score += 0.5

    normalized_field = field_name.strip().lower()
    if normalized_field in GENERIC_FIELD_NAMES:
        score -= 1.0
    else:
        score += min(len(field_name), 20) / 10.0

    if value_page_freq == 1:
        score += 2.0
    else:
        score -= min(value_page_freq, 10) / 2.0

    return score


def build_benchmark(min_value_length: int, max_query_value_chars: int) -> dict[str, Any]:
    page_manifest_path = PROCESSED_ROOT / "page_manifest.jsonl"
    fields_path = PROCESSED_ROOT / "field_annotations.jsonl"
    if not page_manifest_path.exists() or not fields_path.exists():
        raise FileNotFoundError(
            "Missing processed corpus files. Run `python scripts/build_zh_corpus_manifest.py` first."
        )

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    pages = _read_jsonl(page_manifest_path)
    fields = _read_jsonl(fields_path)
    page_by_id = {page["page_id"]: page for page in pages}

    value_pages: dict[str, set[str]] = defaultdict(set)
    for field in fields:
        value = _normalize_text(field["field_value"])
        if _has_signal(value, min_value_length):
            value_pages[value].add(field["page_id"])

    candidates: list[QueryCandidate] = []
    duplicate_query_counter: Counter[str] = Counter()
    field_counter: Counter[str] = Counter()

    for field in fields:
        page = page_by_id.get(field["page_id"])
        if page is None:
            continue

        field_value = _normalize_text(field["field_value"])
        if not _has_signal(field_value, min_value_length):
            continue

        field_name = _normalize_text(field["field_name"])
        query_value = _shorten_value(field_value, max_query_value_chars)
        query = _make_query(field_name, query_value)
        duplicate_query_counter[query] += 1
        field_counter[field_name] += 1
        candidates.append(
            QueryCandidate(
                query_id="",
                query=query,
                page_id=field["page_id"],
                dataset=field["dataset"],
                field_name=field_name,
                field_value=field_value,
                query_value=query_value,
                value_length=len(field_value),
                score=_candidate_score(field_name, field_value, len(value_pages[field_value])),
            )
        )

    unique_candidates = [candidate for candidate in candidates if duplicate_query_counter[candidate.query] == 1]
    unique_candidates.sort(key=lambda item: (item.dataset, item.page_id, -item.score, item.field_name, item.query_value))

    selected_by_page: dict[str, QueryCandidate] = {}
    for candidate in unique_candidates:
        current = selected_by_page.get(candidate.page_id)
        if current is None or candidate.score > current.score:
            selected_by_page[candidate.page_id] = candidate

    selected_candidates = sorted(selected_by_page.values(), key=lambda item: (item.dataset, item.page_id))
    for idx, candidate in enumerate(selected_candidates):
        candidate.query_id = f"zhq_{idx:06d}"

    candidate_rows = [asdict(candidate) for candidate in unique_candidates]
    selected_rows = [asdict(candidate) for candidate in selected_candidates]
    _write_jsonl(BENCHMARK_ROOT / "query_candidates.jsonl", candidate_rows)
    _write_jsonl(BENCHMARK_ROOT / "selected_queries.jsonl", selected_rows)

    qrels = {candidate.query: {candidate.page_id: 1} for candidate in selected_candidates}
    (BENCHMARK_ROOT / "qrels.json").write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")

    queries = [{"query_id": candidate.query_id, "query": candidate.query, "page_id": candidate.page_id} for candidate in selected_candidates]
    _write_jsonl(BENCHMARK_ROOT / "queries.jsonl", queries)

    corpus_rows = [
        {
            "doc_id": page["page_id"],
            "image": page["image_path"],
            "image_filename": page["page_id"],
            "dataset": page["dataset"],
            "source_json_path": page["json_path"],
        }
        for page in pages
    ]
    _write_jsonl(BENCHMARK_ROOT / "corpus.jsonl", corpus_rows)

    rows: list[BenchmarkRow] = []
    for page in pages:
        candidate = selected_by_page.get(page["page_id"])
        rows.append(
            BenchmarkRow(
                query=candidate.query if candidate else None,
                image=page["image_path"],
                image_filename=page["page_id"],
                page_id=page["page_id"],
                dataset=page["dataset"],
                source_image_path=page["image_path"],
                source_json_path=page["json_path"],
                field_name=candidate.field_name if candidate else None,
                field_value=candidate.field_value if candidate else None,
                query_id=candidate.query_id if candidate else None,
            )
        )
    _write_jsonl(BENCHMARK_ROOT / "vidore_compat_test.jsonl", [asdict(row) for row in rows])

    selected_by_dataset: Counter[str] = Counter(candidate.dataset for candidate in selected_candidates)
    candidate_by_dataset: Counter[str] = Counter(candidate.dataset for candidate in unique_candidates)
    report = {
        "filters": {
            "min_value_length": min_value_length,
            "max_query_value_chars": max_query_value_chars,
            "duplicate_query_policy": "drop duplicate query strings before selecting one query per page",
        },
        "counts": {
            "pages": len(pages),
            "raw_fields": len(fields),
            "candidates_before_duplicate_filter": len(candidates),
            "unique_query_candidates": len(unique_candidates),
            "selected_queries": len(selected_candidates),
            "pages_without_query": len(pages) - len(selected_candidates),
        },
        "candidate_by_dataset": dict(candidate_by_dataset),
        "selected_by_dataset": dict(selected_by_dataset),
        "top_candidate_fields": [
            {"field_name": field_name, "count": count} for field_name, count in field_counter.most_common(50)
        ],
        "outputs": {
            "query_candidates": str((BENCHMARK_ROOT / "query_candidates.jsonl").resolve()),
            "selected_queries": str((BENCHMARK_ROOT / "selected_queries.jsonl").resolve()),
            "qrels": str((BENCHMARK_ROOT / "qrels.json").resolve()),
            "queries": str((BENCHMARK_ROOT / "queries.jsonl").resolve()),
            "corpus": str((BENCHMARK_ROOT / "corpus.jsonl").resolve()),
            "vidore_compat_test": str((BENCHMARK_ROOT / "vidore_compat_test.jsonl").resolve()),
        },
    }
    (REPORTS_ROOT / "benchmark_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated local Chinese ViDoRe-compatible benchmark draft.")
    parser.add_argument("--min-value-length", type=int, default=10)
    parser.add_argument("--max-query-value-chars", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_benchmark(
        min_value_length=args.min_value_length,
        max_query_value_chars=args.max_query_value_chars,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
