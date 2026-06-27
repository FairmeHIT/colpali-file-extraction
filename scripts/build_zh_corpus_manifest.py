#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


RAW_ROOT = ROOT_DIR / "local_data" / "zh_corpus" / "raw"
PROCESSED_ROOT = ROOT_DIR / "local_data" / "zh_corpus" / "processed"
REPORTS_ROOT = ROOT_DIR / "local_data" / "zh_corpus" / "reports"

SUPPORTED_DATASETS = ["XFUND-zh-val", "COLD_CELL_600", "EPHOIE_SCUT_311"]


@dataclass
class PageRecord:
    dataset: str
    page_id: str
    image_path: str
    json_path: str
    width: int | None = None
    height: int | None = None
    image_name: str | None = None
    query_source: str | None = None


@dataclass
class FieldRecord:
    dataset: str
    page_id: str
    field_name: str
    field_value: str
    source: str
    query_hint: str


def _ensure_dirs() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)


def _iter_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_dirs = [dataset_dir / "images", dataset_dir / "validataion_images_ok", dataset_dir / "validation_images_ok"]
    primary_json_dir = dataset_dir / "json"
    fallback_json_dirs = [dataset_dir / "json_old"]

    image_map: dict[str, Path] = {}
    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                image_map[p.stem] = p

    pairs_by_stem: dict[str, tuple[Path, Path]] = {}
    if primary_json_dir.exists():
        for p in primary_json_dir.rglob("*.json"):
            img = image_map.get(p.stem)
            if img is not None:
                pairs_by_stem[p.stem] = (img, p)

    for js_dir in fallback_json_dirs:
        if not js_dir.exists():
            continue
        for p in js_dir.rglob("*.json"):
            if p.stem in pairs_by_stem:
                continue
            img = image_map.get(p.stem)
            if img is not None:
                pairs_by_stem[p.stem] = (img, p)
    return sorted(pairs_by_stem.values(), key=lambda x: x[1].name)


def _extract_fields(obj: dict[str, Any], dataset: str) -> list[tuple[str, str]]:
    output = obj.get("output")
    if isinstance(output, dict):
        items: list[tuple[str, str]] = []
        for k, v in output.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (str, int, float)) and str(sv).strip():
                        items.append((f"{k}.{sk}", str(sv).strip()))
            elif isinstance(v, list):
                joined = "；".join(str(x).strip() for x in v if str(x).strip())
                if joined:
                    items.append((k, joined))
            elif isinstance(v, (str, int, float)):
                sv = str(v).strip()
                if sv:
                    items.append((k, sv))
        return items

    if isinstance(output, str):
        value = output.strip()
        return [("answer", value)] if value else []

    return []


def build_manifest() -> None:
    _ensure_dirs()

    page_records: list[PageRecord] = []
    field_records: list[FieldRecord] = []
    summary: dict[str, Any] = {"datasets": {}}
    field_name_counter: Counter[str] = Counter()
    field_value_lengths: list[int] = []

    for ds in SUPPORTED_DATASETS:
        ds_dir = RAW_ROOT / ds
        pairs = _iter_pairs(ds_dir)
        ds_summary = {"pages": len(pairs), "fields": 0}

        for img_path, json_path in pairs:
            page_id = f"{ds}/{img_path.stem}"
            width = None
            height = None
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                pass

            page_record = PageRecord(
                dataset=ds,
                page_id=page_id,
                image_path=str(img_path.resolve()),
                json_path=str(json_path.resolve()),
                width=width,
                height=height,
                image_name=img_path.name,
                query_source="json.output",
            )
            page_records.append(page_record)

            try:
                obj = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            fields = _extract_fields(obj, ds)
            ds_summary["fields"] += len(fields)
            for field_name, field_value in fields:
                field_name_counter[field_name] += 1
                field_value_lengths.append(len(field_value))
                hint = f"请找到包含字段“{field_name}”且答案为“{field_value[:32]}”的页面。"
                field_records.append(
                    FieldRecord(
                        dataset=ds,
                        page_id=page_id,
                        field_name=field_name,
                        field_value=field_value,
                        source="json.output",
                        query_hint=hint,
                    )
                )

        summary["datasets"][ds] = ds_summary

    (PROCESSED_ROOT / "page_manifest.jsonl").write_text(
        "\n".join(json.dumps(asdict(x), ensure_ascii=False) for x in page_records) + "\n",
        encoding="utf-8",
    )
    (PROCESSED_ROOT / "field_annotations.jsonl").write_text(
        "\n".join(json.dumps(asdict(x), ensure_ascii=False) for x in field_records) + "\n",
        encoding="utf-8",
    )
    summary["totals"] = {"pages": len(page_records), "fields": len(field_records)}
    if field_value_lengths:
        sorted_lengths = sorted(field_value_lengths)
        summary["field_value_lengths"] = {
            "min": sorted_lengths[0],
            "median": sorted_lengths[len(sorted_lengths) // 2],
            "max": sorted_lengths[-1],
            "short_le_2": sum(1 for x in sorted_lengths if x <= 2),
        }
    summary["top_field_names"] = [
        {"field_name": field_name, "count": count} for field_name, count in field_name_counter.most_common(50)
    ]
    (REPORTS_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_manifest()
