#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


CORPUS_ROOT = ROOT_DIR / "local_data" / "zh_corpus"
PROCESSED_ROOT = CORPUS_ROOT / "processed"
BENCHMARK_ROOT = CORPUS_ROOT / "benchmark"
MINERU_ROOT = CORPUS_ROOT / "mineru"
REPORTS_ROOT = CORPUS_ROOT / "reports"


@dataclass
class MinerUPageResult:
    page_id: str
    dataset: str
    image_path: str
    mode: str
    state: str
    task_id: str | None
    filename: str | None
    err_code: str
    error: str | None
    markdown_path: str | None
    json_path: str
    markdown_chars: int
    elapsed_sec: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_page_id(page_id: str) -> str:
    return page_id.replace("/", "__").replace("\\", "__")


def _load_pages(scope: str) -> list[dict[str, Any]]:
    page_manifest_path = PROCESSED_ROOT / "page_manifest.jsonl"
    if not page_manifest_path.exists():
        raise FileNotFoundError("Missing page manifest. Run `python scripts/build_zh_corpus_manifest.py` first.")

    pages = _read_jsonl(page_manifest_path)
    if scope == "all":
        return pages

    selected_path = BENCHMARK_ROOT / "selected_queries.jsonl"
    if not selected_path.exists():
        raise FileNotFoundError("Missing selected queries. Run `python scripts/build_zh_local_benchmark.py` first.")
    selected_page_ids = {row["page_id"] for row in _read_jsonl(selected_path)}
    return [page for page in pages if page["page_id"] in selected_page_ids]


def _result_to_jsonable(result: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in [
        "task_id",
        "state",
        "filename",
        "err_code",
        "error",
        "zip_url",
        "markdown",
        "content_list",
        "html",
        "latex",
    ]:
        value = getattr(result, key, None)
        if value is not None:
            data[key] = value
    progress = getattr(result, "progress", None)
    if progress is not None:
        try:
            data["progress"] = asdict(progress)
        except TypeError:
            data["progress"] = str(progress)
    images = getattr(result, "images", None)
    if images:
        data["images"] = [{"name": getattr(img, "name", None), "path": getattr(img, "path", None)} for img in images]
    return data


def _call_mineru(
    client: Any,
    image_path: str,
    mode: str,
    timeout: int,
    ocr: bool,
    table: bool,
    model: str | None,
) -> Any:
    if mode == "flash":
        return client.flash_extract(
            image_path,
            language="ch",
            is_ocr=ocr,
            enable_table=table,
            timeout=timeout,
        )
    return client.extract(
        image_path,
        model=model,
        ocr=ocr,
        table=table,
        language="ch",
        timeout=timeout,
    )


def enrich_with_mineru(
    scope: str,
    mode: str,
    limit: int | None,
    timeout: int,
    sleep_sec: float,
    overwrite: bool,
    ocr: bool,
    table: bool,
    model: str | None,
    no_proxy: bool,
) -> dict[str, Any]:
    from mineru import MinerU

    if no_proxy:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(key, None)

    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("MINERU_TOKEN")
    if mode == "extract" and not token:
        raise RuntimeError("MinerU extract mode requires MINERU_API_TOKEN or MINERU_TOKEN.")

    pages = _load_pages(scope)
    if limit is not None:
        pages = pages[:limit]

    output_dir = MINERU_ROOT / mode
    raw_dir = output_dir / "raw"
    markdown_dir = output_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    client = MinerU(
        token=token,
        base_url=os.environ.get("MINERU_BASE_URL", "https://mineru.net/api/v4"),
        flash_base_url=os.environ.get("MINERU_FLASH_BASE_URL"),
    )
    results: list[MinerUPageResult] = []
    failed = 0
    skipped = 0

    try:
        for idx, page in enumerate(pages, start=1):
            page_id = page["page_id"]
            safe_id = _safe_page_id(page_id)
            raw_path = raw_dir / f"{safe_id}.json"
            markdown_path = markdown_dir / f"{safe_id}.md"

            if raw_path.exists() and not overwrite:
                skipped += 1
                continue

            started = time.monotonic()
            print(f"[{idx}/{len(pages)}] MinerU {mode}: {page_id}", flush=True)
            try:
                result = _call_mineru(
                    client=client,
                    image_path=page["image_path"],
                    mode=mode,
                    timeout=timeout,
                    ocr=ocr,
                    table=table,
                    model=model,
                )
                raw_data = _result_to_jsonable(result)
                raw_data["page_id"] = page_id
                raw_data["dataset"] = page["dataset"]
                raw_data["image_path"] = page["image_path"]
                raw_data["source_json_path"] = page["json_path"]
                markdown = raw_data.get("markdown") or ""
                if markdown:
                    markdown_path.write_text(markdown, encoding="utf-8")
                raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")

                results.append(
                    MinerUPageResult(
                        page_id=page_id,
                        dataset=page["dataset"],
                        image_path=page["image_path"],
                        mode=mode,
                        state=raw_data.get("state", ""),
                        task_id=raw_data.get("task_id"),
                        filename=raw_data.get("filename"),
                        err_code=raw_data.get("err_code", ""),
                        error=raw_data.get("error"),
                        markdown_path=str(markdown_path.resolve()) if markdown else None,
                        json_path=str(raw_path.resolve()),
                        markdown_chars=len(markdown),
                        elapsed_sec=round(time.monotonic() - started, 3),
                    )
                )
            except Exception as exc:
                failed += 1
                error_data = {
                    "page_id": page_id,
                    "dataset": page["dataset"],
                    "image_path": page["image_path"],
                    "mode": mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                raw_path.write_text(json.dumps(error_data, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(
                    MinerUPageResult(
                        page_id=page_id,
                        dataset=page["dataset"],
                        image_path=page["image_path"],
                        mode=mode,
                        state="failed",
                        task_id=None,
                        filename=None,
                        err_code=type(exc).__name__,
                        error=str(exc),
                        markdown_path=None,
                        json_path=str(raw_path.resolve()),
                        markdown_chars=0,
                        elapsed_sec=round(time.monotonic() - started, 3),
                    )
                )

            if sleep_sec > 0 and idx < len(pages):
                time.sleep(sleep_sec)
    finally:
        client.close()

    index_path = output_dir / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    report = {
        "mode": mode,
        "scope": scope,
        "limit": limit,
        "timeout": timeout,
        "ocr": ocr,
        "table": table,
        "model": model,
        "no_proxy": no_proxy,
        "total_requested": len(pages),
        "processed_this_run": len(results),
        "skipped_existing": skipped,
        "failed_this_run": failed,
        "index_path": str(index_path.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "markdown_dir": str(markdown_dir.resolve()),
    }
    report_path = REPORTS_ROOT / f"mineru_{mode}_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich the isolated Chinese corpus with official MinerU API output.")
    parser.add_argument("--scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--mode", choices=["flash", "extract"], default="flash")
    parser.add_argument("--limit", type=int, default=5, help="Default is a small smoke batch. Use 0 for no limit.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--model", default=None, help="Only used by extract mode, for example pipeline or vlm.")
    parser.add_argument("--no-proxy", action="store_true", help="Unset HTTP(S)_PROXY for MinerU API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    report = enrich_with_mineru(
        scope=args.scope,
        mode=args.mode,
        limit=limit,
        timeout=args.timeout,
        sleep_sec=args.sleep_sec,
        overwrite=args.overwrite,
        ocr=not args.no_ocr,
        table=not args.no_table,
        model=args.model,
        no_proxy=args.no_proxy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
