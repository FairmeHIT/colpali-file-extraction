#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, Image
from mteb.evaluation.evaluators import RetrievalEvaluator


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
    elif model_name == "BAAI/bge-m3":
        import vidore_benchmark.retrievers.bge_m3_retriever  # noqa: F401
    elif model_name == "BAAI/bge-m3-colbert":
        import vidore_benchmark.retrievers.bge_m3_colbert_retriever  # noqa: F401


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize_zh_mixed(text: Any) -> list[str]:
    text = str(text or "").lower()
    tokens: list[str] = []
    for word in re.findall(r"[a-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text):
        tokens.append(word)
    for word in re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", text):
        if re.search(r"\d", word) or re.search(r"[a-z]", word):
            tokens.append(word)
    return tokens


class BM25Index:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_len = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for doc in tokenized_docs:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            self.doc_freq.update(tf.keys())
        self.n_docs = len(tokenized_docs)
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0 or not query_tokens:
            return scores
        query_terms = Counter(query_tokens)
        for term, qtf in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_len[idx] / max(self.avgdl, 1e-6))
                scores[idx] += idf * (freq * (self.k1 + 1.0) / denom) * qtf
        return scores


def load_rows(path: Path, limit_queries: int, include_unqueried: bool) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not include_unqueried:
        rows = [row for row in rows if row.get("query")]
    if limit_queries > 0:
        query_count = 0
        limited: list[dict[str, Any]] = []
        for row in rows:
            limited.append(row)
            if row.get("query"):
                query_count += 1
            if query_count >= limit_queries:
                break
        rows = limited
    if not rows:
        raise ValueError(f"No rows available after filtering `{path}`.")
    missing_images = [row["image"] for row in rows if not Path(row["image"]).exists()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} image paths are missing. First: {missing_images[0]}")
    return rows


def make_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows).cast_column("image", Image())


def get_unique_queries(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    queries: list[str] = []
    for row in rows:
        query = row.get("query")
        if query and query not in seen:
            queries.append(query)
            seen.add(query)
    return queries


def relevant_docs_from_rows(rows: list[dict[str, Any]], queries: list[str]) -> dict[str, dict[str, int]]:
    query_to_filename = {row["query"]: row["image_filename"] for row in rows if row.get("query")}
    return {query: {query_to_filename[query]: 1} for query in queries}


def bm25_scores(rows: list[dict[str, Any]], queries: list[str]) -> np.ndarray:
    docs = [row.get("text_description") or "" for row in rows]
    index = BM25Index([tokenize_zh_mixed(doc) for doc in docs])
    scores = [index.get_scores(tokenize_zh_mixed(query)) for query in queries]
    return np.vstack(scores) if scores else np.zeros((0, len(rows)), dtype=np.float32)


def colpali_scores(rows: list[dict[str, Any]], queries: list[str], model_name: str, batch_query: int, batch_doc: int, batch_score: int) -> np.ndarray:
    from vidore_benchmark.retrievers.utils.load_retriever import load_vision_retriever_from_registry

    import_retriever_for_model(model_name)
    retriever = load_vision_retriever_from_registry(model_name)()
    dataset = make_dataset(rows)
    documents = list(dataset["image"]) if retriever.use_visual_embedding else list(dataset["text_description"])
    emb_queries = retriever.forward_queries(queries, batch_size=batch_query)
    emb_documents = retriever.forward_documents(documents, batch_size=batch_doc)
    scores = retriever.get_scores(emb_queries, emb_documents, batch_size=batch_score)
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().numpy()
    return np.asarray(scores, dtype=np.float32)


def rankdata_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    for row_idx in range(scores.shape[0]):
        ranks[row_idx, order[row_idx]] = np.arange(1, scores.shape[1] + 1, dtype=np.float32)
    return ranks


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    return (scores - mins) / denom


def fuse_scores(colpali: np.ndarray | None, bm25: np.ndarray | None, mode: str, alpha: float, rrf_k: int) -> np.ndarray:
    if mode == "bm25":
        if bm25 is None:
            raise ValueError("BM25 scores are required.")
        return bm25
    if mode == "colpali":
        if colpali is None:
            raise ValueError("ColPali scores are required.")
        return colpali
    if colpali is None or bm25 is None:
        raise ValueError("Hybrid modes require both ColPali and BM25 scores.")
    if mode == "linear":
        return alpha * minmax_normalize(colpali) + (1.0 - alpha) * minmax_normalize(bm25)
    if mode == "rrf":
        colpali_ranks = rankdata_desc(colpali)
        bm25_ranks = rankdata_desc(bm25)
        return 1.0 / (rrf_k + colpali_ranks) + 1.0 / (rrf_k + bm25_ranks)
    raise ValueError(f"Unsupported mode: {mode}")


def scores_to_results(scores: np.ndarray, queries: list[str], doc_ids: list[str]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for qidx, query in enumerate(queries):
        results[query] = {doc_id: float(scores[qidx, didx]) for didx, doc_id in enumerate(doc_ids)}
    return results


def compute_metrics(relevant_docs: dict[str, dict[str, int]], results: dict[str, dict[str, float]]) -> dict[str, float]:
    evaluator = RetrievalEvaluator()
    ndcg, _map, recall, precision, naucs = evaluator.evaluate(
        relevant_docs,
        results,
        evaluator.k_values,
        ignore_identical_ids=True,
    )
    mrr = evaluator.evaluate_custom(relevant_docs, results, evaluator.k_values, "mrr")
    return {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
        **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BM25/ColPali hybrid on local zh_corpus MinerU benchmark.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-label", default="zh_corpus_mineru")
    parser.add_argument("--mode", choices=["bm25", "colpali", "rrf", "linear"], default="rrf")
    parser.add_argument("--model-name", default="vidore/colpali")
    parser.add_argument("--batch-query", type=int, default=1)
    parser.add_argument("--batch-doc", type=int, default=1)
    parser.add_argument("--batch-score", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N query rows for a smoke run. 0 means all.")
    parser.add_argument("--include-unqueried", action="store_true", help="Keep rows with query=None in the document pool.")
    parser.add_argument("--alpha", type=float, default=0.5, help="ColPali weight for linear fusion.")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "outputs" / "zh_corpus_mineru")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path if args.dataset_path.is_absolute() else ROOT_DIR / args.dataset_path
    rows = load_rows(dataset_path, limit_queries=args.limit, include_unqueried=args.include_unqueried)
    queries = get_unique_queries(rows)
    doc_ids = [row["image_filename"] for row in rows]
    relevant_docs = relevant_docs_from_rows(rows, queries)

    print(
        f"Loaded rows={len(rows)}, queries={len(queries)}, docs={len(doc_ids)}, mode={args.mode}",
        flush=True,
    )

    bm25 = bm25_scores(rows, queries) if args.mode in {"bm25", "rrf", "linear"} else None
    colpali = (
        colpali_scores(
            rows,
            queries,
            model_name=args.model_name,
            batch_query=args.batch_query,
            batch_doc=args.batch_doc,
            batch_score=args.batch_score,
        )
        if args.mode in {"colpali", "rrf", "linear"}
        else None
    )
    final_scores = fuse_scores(colpali, bm25, mode=args.mode, alpha=args.alpha, rrf_k=args.rrf_k)
    results = scores_to_results(final_scores, queries, doc_ids)
    metrics = compute_metrics(relevant_docs, results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit_{args.limit}" if args.limit > 0 else ""
    model_slug = "bm25" if args.mode == "bm25" else args.model_name.replace("/", "_")
    output_path = args.output_dir / f"{model_slug}_{args.dataset_label}_{args.mode}{suffix}_metrics.json"
    payload = {
        args.dataset_label: {
            "mode": args.mode,
            "model_name": args.model_name if args.mode != "bm25" else None,
            "alpha": args.alpha if args.mode == "linear" else None,
            "rrf_k": args.rrf_k if args.mode == "rrf" else None,
            "rows": len(rows),
            "queries": len(queries),
            "docs": len(doc_ids),
            "metrics": metrics,
        }
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metrics saved to `{output_path}`", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
