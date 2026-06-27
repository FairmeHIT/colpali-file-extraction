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
MINERU_ROOT = CORPUS_ROOT / "mineru" / "flash"
BENCHMARK_ROOT = CORPUS_ROOT / "benchmark_mineru"
REPORTS_ROOT = CORPUS_ROOT / "reports"

BOILERPLATE_PATTERNS = [
    r"<!--\s*image\s*-->",
    r"rowspan=\d+",
    r"colspan=\d+",
    r"</?table[^>]*>",
    r"</?tr[^>]*>",
    r"</?td[^>]*>",
    r"</?th[^>]*>",
    r"^\s*第\s*\d+\s*页\s*共\s*\d+\s*页\s*$",
]

WEAK_LINE_PATTERNS = [
    r"^[\d\s:：/\\\-年月日.]+$",
    r"^[A-Za-z0-9\s:：/\\\-_.]+$",
    r"^(备注|注|合计|小计|金额|电话|日期|时间|编号|页码)$",
]

DATE_RE = re.compile(r"\d{4}[年/\-.]\d{1,2}[月/\-.]\d{1,2}日?")
PHONE_RE = re.compile(r"(?:1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8})")
MONEY_RE = re.compile(r"(?:RMB[:：]?\s*)?\d+(?:\.\d{1,2})?\s*(?:元|￥|¥|RMB)")
ID_RE = re.compile(r"[A-Z0-9]{2,}[-A-Z0-9]{4,}")


@dataclass
class MinerUPageText:
    page_id: str
    dataset: str
    image_path: str
    source_json_path: str
    markdown_path: str
    raw_markdown_chars: int
    clean_text: str
    clean_text_chars: int


@dataclass
class QueryCandidate:
    query_id: str
    query: str
    page_id: str
    dataset: str
    source: str
    evidence: str
    score: float
    evidence_length: int


@dataclass
class BenchmarkRow:
    query: str | None
    image: str
    image_filename: str
    page_id: str
    dataset: str
    source_image_path: str
    source_json_path: str
    text_description: str | None = None
    mineru_markdown_path: str | None = None
    query_id: str | None = None
    evidence: str | None = None
    query_source: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _safe_page_id(page_id: str) -> str:
    return page_id.replace("/", "__").replace("\\", "__")


def _normalize_text(value: str) -> str:
    value = re.sub(r"\\\s*", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _strip_markdown(markdown: str) -> str:
    text = markdown
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[#*_`]+", " ", text)
    text = text.replace("|", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\r", "\n", text)
    return text


def _iter_lines(markdown: str) -> list[str]:
    text = _strip_markdown(markdown)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_text(raw_line)
        if not line:
            continue
        line = line.strip(" ，,。；;:：、-")
        if len(line) < 4:
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in WEAK_LINE_PATTERNS):
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
            continue
        lines.append(line)
    return lines


def _clean_text(markdown: str, max_chars: int) -> str:
    text = " ".join(_iter_lines(markdown))
    text = _normalize_text(text)
    return text[:max_chars]


def _shorten(value: str, max_chars: int) -> str:
    value = _normalize_text(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip("，,。；;、 ") + "..."


def _looks_unique(value: str) -> bool:
    if len(value) >= 12:
        return True
    return bool(DATE_RE.search(value) or PHONE_RE.search(value) or MONEY_RE.search(value) or ID_RE.search(value))


def _line_score(line: str, line_idx: int, page_freq: int) -> float:
    score = 0.0
    length = len(line)
    if 10 <= length <= 80:
        score += 4.0
    elif 6 <= length < 10:
        score += 2.0
    elif 80 < length <= 160:
        score += 2.5
    else:
        score += 1.0

    if re.search(r"[\u4e00-\u9fff]", line):
        score += 2.0
    if re.search(r"\d", line):
        score += 1.0
    if any(regex.search(line) for regex in [DATE_RE, PHONE_RE, MONEY_RE, ID_RE]):
        score += 1.5
    if line_idx <= 5:
        score += 1.0
    if page_freq == 1:
        score += 2.0
    else:
        score -= min(page_freq, 10) / 2.0
    return score


def _evidence_focus(source: str, evidence: str) -> str:
    if source == "date":
        match = DATE_RE.search(evidence)
        if match:
            return match.group(0)
    if source == "phone":
        match = PHONE_RE.search(evidence)
        if match:
            return match.group(0)
    if source == "money":
        match = MONEY_RE.search(evidence)
        if match:
            return match.group(0)
    if source == "identifier":
        if "：" in evidence or ":" in evidence:
            return evidence
        match = ID_RE.search(evidence)
        if match:
            return match.group(0)
    return evidence


def _make_query(source: str, evidence: str, max_chars: int) -> str:
    evidence = _shorten(_evidence_focus(source, evidence), max_chars)
    if source == "title":
        return f"查找标题或抬头包含“{evidence}”的中文文档页面。"
    if source == "date":
        return f"查找包含日期“{evidence}”的中文文档页面。"
    if source == "phone":
        return f"查找包含联系电话“{evidence}”的中文文档页面。"
    if source == "money":
        return f"查找包含金额“{evidence}”的中文文档页面。"
    if source == "identifier":
        return f"查找包含编号“{evidence}”的中文文档页面。"
    return f"查找包含文本“{evidence}”的中文文档页面。"


def _candidate_source(line: str, line_idx: int) -> str:
    if DATE_RE.search(line):
        return "date"
    if PHONE_RE.search(line):
        return "phone"
    if MONEY_RE.search(line):
        return "money"
    if ID_RE.search(line):
        return "identifier"
    if line_idx <= 3 and len(line) <= 40:
        return "title"
    return "text"


def _load_page_texts(mode: str, clean_text_max_chars: int) -> list[MinerUPageText]:
    page_manifest_path = PROCESSED_ROOT / "page_manifest.jsonl"
    if not page_manifest_path.exists():
        raise FileNotFoundError("Missing page manifest. Run `python scripts/build_zh_corpus_manifest.py` first.")

    pages = _read_jsonl(page_manifest_path)
    page_by_id = {page["page_id"]: page for page in pages}

    index_path = MINERU_ROOT / "index.jsonl"
    markdown_by_page: dict[str, Path] = {}
    if index_path.exists():
        for row in _read_jsonl(index_path):
            markdown_path = row.get("markdown_path")
            page_id = row.get("page_id")
            if page_id and markdown_path and Path(markdown_path).exists():
                markdown_by_page[page_id] = Path(markdown_path)

    markdown_dir = MINERU_ROOT / "markdown"
    if markdown_dir.exists():
        for path in markdown_dir.glob("*.md"):
            page_id = path.stem.replace("__", "/")
            if page_id in page_by_id:
                markdown_by_page.setdefault(page_id, path)

    page_texts: list[MinerUPageText] = []
    for page_id, markdown_path in sorted(markdown_by_page.items()):
        page = page_by_id.get(page_id)
        if page is None:
            continue
        markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
        clean_text = _clean_text(markdown, clean_text_max_chars)
        if mode == "with_text" and not clean_text:
            continue
        page_texts.append(
            MinerUPageText(
                page_id=page_id,
                dataset=page["dataset"],
                image_path=page["image_path"],
                source_json_path=page["json_path"],
                markdown_path=str(markdown_path.resolve()),
                raw_markdown_chars=len(markdown),
                clean_text=clean_text,
                clean_text_chars=len(clean_text),
            )
        )
    return page_texts


def build_benchmark(
    min_evidence_chars: int,
    max_evidence_chars: int,
    clean_text_max_chars: int,
    min_score: float,
) -> dict[str, Any]:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    page_texts = _load_page_texts(mode="with_text", clean_text_max_chars=clean_text_max_chars)
    lines_by_page: dict[str, list[str]] = {}
    line_pages: dict[str, set[str]] = defaultdict(set)

    for page in page_texts:
        markdown = Path(page.markdown_path).read_text(encoding="utf-8", errors="ignore")
        lines = []
        seen = set()
        for line in _iter_lines(markdown):
            if len(line) < min_evidence_chars or len(line) > max_evidence_chars:
                continue
            if not _looks_unique(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            line_pages[line].add(page.page_id)
        lines_by_page[page.page_id] = lines

    candidates: list[QueryCandidate] = []
    duplicate_query_counter: Counter[str] = Counter()
    for page in page_texts:
        for idx, line in enumerate(lines_by_page.get(page.page_id, [])):
            source = _candidate_source(line, idx)
            query = _make_query(source, line, max_evidence_chars)
            duplicate_query_counter[query] += 1
            candidates.append(
                QueryCandidate(
                    query_id="",
                    query=query,
                    page_id=page.page_id,
                    dataset=page.dataset,
                    source=source,
                    evidence=line,
                    score=_line_score(line, idx, len(line_pages[line])),
                    evidence_length=len(line),
                )
            )

    unique_candidates = [
        candidate
        for candidate in candidates
        if duplicate_query_counter[candidate.query] == 1 and candidate.score >= min_score
    ]
    unique_candidates.sort(key=lambda item: (item.dataset, item.page_id, -item.score, item.source, item.evidence))

    selected_by_page: dict[str, QueryCandidate] = {}
    for candidate in unique_candidates:
        current = selected_by_page.get(candidate.page_id)
        if current is None or candidate.score > current.score:
            selected_by_page[candidate.page_id] = candidate

    selected_candidates = sorted(selected_by_page.values(), key=lambda item: (item.dataset, item.page_id))
    for idx, candidate in enumerate(selected_candidates):
        candidate.query_id = f"zhm_{idx:06d}"

    page_by_id = {page.page_id: page for page in page_texts}
    _write_jsonl(BENCHMARK_ROOT / "mineru_page_texts.jsonl", [asdict(page) for page in page_texts])
    _write_jsonl(BENCHMARK_ROOT / "query_candidates.jsonl", [asdict(candidate) for candidate in unique_candidates])
    _write_jsonl(BENCHMARK_ROOT / "selected_queries.jsonl", [asdict(candidate) for candidate in selected_candidates])

    qrels = {candidate.query: {candidate.page_id: 1} for candidate in selected_candidates}
    (BENCHMARK_ROOT / "qrels.json").write_text(json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8")

    queries = [
        {
            "query_id": candidate.query_id,
            "query": candidate.query,
            "page_id": candidate.page_id,
            "evidence": candidate.evidence,
            "source": candidate.source,
        }
        for candidate in selected_candidates
    ]
    _write_jsonl(BENCHMARK_ROOT / "queries.jsonl", queries)

    corpus_rows = [
        {
            "doc_id": page.page_id,
            "image": page.image_path,
            "image_filename": page.page_id,
            "dataset": page.dataset,
            "text_description": page.clean_text,
            "mineru_markdown_path": page.markdown_path,
            "source_json_path": page.source_json_path,
        }
        for page in page_texts
    ]
    _write_jsonl(BENCHMARK_ROOT / "corpus.jsonl", corpus_rows)

    rows: list[BenchmarkRow] = []
    for page in page_texts:
        candidate = selected_by_page.get(page.page_id)
        rows.append(
            BenchmarkRow(
                query=candidate.query if candidate else None,
                image=page.image_path,
                image_filename=page.page_id,
                page_id=page.page_id,
                dataset=page.dataset,
                source_image_path=page.image_path,
                source_json_path=page.source_json_path,
                text_description=page.clean_text,
                mineru_markdown_path=page.markdown_path,
                query_id=candidate.query_id if candidate else None,
                evidence=candidate.evidence if candidate else None,
                query_source=candidate.source if candidate else None,
            )
        )
    _write_jsonl(BENCHMARK_ROOT / "vidore_compat_test.jsonl", [asdict(row) for row in rows])

    selected_by_dataset: Counter[str] = Counter(candidate.dataset for candidate in selected_candidates)
    candidate_by_dataset: Counter[str] = Counter(candidate.dataset for candidate in unique_candidates)
    selected_by_source: Counter[str] = Counter(candidate.source for candidate in selected_candidates)
    report = {
        "source": "MinerU flash markdown",
        "filters": {
            "min_evidence_chars": min_evidence_chars,
            "max_evidence_chars": max_evidence_chars,
            "clean_text_max_chars": clean_text_max_chars,
            "min_score": min_score,
            "duplicate_query_policy": "drop duplicate query strings before selecting one query per page",
        },
        "counts": {
            "mineru_pages_with_text": len(page_texts),
            "candidates_before_duplicate_filter": len(candidates),
            "unique_query_candidates": len(unique_candidates),
            "selected_queries": len(selected_candidates),
            "pages_without_query": len(page_texts) - len(selected_candidates),
        },
        "candidate_by_dataset": dict(candidate_by_dataset),
        "selected_by_dataset": dict(selected_by_dataset),
        "selected_by_source": dict(selected_by_source),
        "outputs": {
            "mineru_page_texts": str((BENCHMARK_ROOT / "mineru_page_texts.jsonl").resolve()),
            "query_candidates": str((BENCHMARK_ROOT / "query_candidates.jsonl").resolve()),
            "selected_queries": str((BENCHMARK_ROOT / "selected_queries.jsonl").resolve()),
            "qrels": str((BENCHMARK_ROOT / "qrels.json").resolve()),
            "queries": str((BENCHMARK_ROOT / "queries.jsonl").resolve()),
            "corpus": str((BENCHMARK_ROOT / "corpus.jsonl").resolve()),
            "vidore_compat_test": str((BENCHMARK_ROOT / "vidore_compat_test.jsonl").resolve()),
        },
    }
    (REPORTS_ROOT / "benchmark_mineru_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MinerU-content Chinese ViDoRe-compatible benchmark.")
    parser.add_argument("--min-evidence-chars", type=int, default=8)
    parser.add_argument("--max-evidence-chars", type=int, default=80)
    parser.add_argument("--clean-text-max-chars", type=int, default=4000)
    parser.add_argument("--min-score", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_benchmark(
        min_evidence_chars=args.min_evidence_chars,
        max_evidence_chars=args.max_evidence_chars,
        clean_text_max_chars=args.clean_text_max_chars,
        min_score=args.min_score,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
