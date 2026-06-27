from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass
class NormalizedSample:
    query_id: str
    query: str
    anchor_doc_id: str
    anchor_image_path: str
    anchor_image_filename: str
    positive_doc_ids: list[str]
    positive_image_paths: list[str]
    answer: Any = None
    source_dataset: str | None = None
    source_record_path: str | None = None
    source_record_id: str | None = None
    evidence_pages: list[int] | None = None
    doc_no: str | None = None
    task_tag: str | None = None
    question_type: str | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_records(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        for key in ("data", "records", "items", "examples", "samples"):
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [obj]
    return []


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def iter_record_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}]
    return sorted(files)


def load_records(source: Path) -> list[dict[str, Any]]:
    files = iter_record_files(source)
    if source.is_file() and source.suffix.lower() in {".json", ".jsonl"}:
        files = [source]
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl_records(path))
        elif path.suffix.lower() == ".json":
            rows.extend(read_json_records(path))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_keys(value: str | None, defaults: list[str]) -> list[str]:
    if not value:
        return defaults
    keys = [normalize_text(item) for item in value.split(",")]
    return [key for key in keys if key]


def pick_first(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def flatten_positive_pages(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        output: list[int] = []
        for item in value:
            output.extend(flatten_positive_pages(item))
        return sorted(set(output))
    return []


def index_images(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not image_root.exists():
        return index
    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(image_root).as_posix()
        index[rel] = path
        index[path.name] = path
        index[path.stem] = path
    return index


def resolve_image_path(value: Any, image_root: Path, image_index: dict[str, Path]) -> Path | None:
    if value is None:
        return None
    candidate = normalize_text(value)
    if not candidate:
        return None

    raw_path = Path(candidate)
    if raw_path.exists():
        return raw_path.resolve()

    rel_candidate = image_root / candidate
    if rel_candidate.exists():
        return rel_candidate.resolve()

    stem = raw_path.stem
    if stem in image_index:
        return image_index[stem].resolve()

    name = raw_path.name
    if name in image_index:
        return image_index[name].resolve()

    rel = candidate.replace("\\", "/")
    if rel in image_index:
        return image_index[rel].resolve()

    return None


def build_corpus_rows(image_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        doc_id = path.relative_to(image_root).as_posix()
        rows.append(
            {
                "doc_id": doc_id,
                "image": str(path.resolve()),
                "image_filename": doc_id,
            }
        )
    return rows

