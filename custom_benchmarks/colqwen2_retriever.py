from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_COLPALI_REPRO = ROOT_DIR / "colpali-repro"


def _prefer_local_colpali_repro() -> None:
    if LOCAL_COLPALI_REPRO.exists() and str(LOCAL_COLPALI_REPRO) not in sys.path:
        sys.path.insert(0, str(LOCAL_COLPALI_REPRO))


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[Any]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


def _torch_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalColQwen2Retriever:
    """Isolated ColQwen2 retriever for local certificate benchmark experiments."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "vidore/colqwen2-v0.1",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> None:
        _prefer_local_colpali_repro()

        try:
            from transformers.models.qwen2_vl import Qwen2VLProcessor  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "ColQwen2 requires transformers with qwen2_vl support. Current .venv-vidore has transformers 4.41.2, "
                "which is too old. Use the isolated .venv-colqwen2 setup instead of upgrading .venv-vidore."
            ) from exc

        try:
            from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import ColQwen2
            from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor
        except Exception as exc:
            raise RuntimeError(
                "ColQwen2 requires a colpali-engine version that exposes ColQwen2/ColQwen2Processor. "
                "Use the isolated ColQwen2 environment command in local_data/zh_cert_benchmark/README.md."
            ) from exc

        try:
            from transformers.utils.import_utils import is_flash_attn_2_available
        except Exception:
            is_flash_attn_2_available = lambda: False

        self.device = _torch_device(device)
        dtype = getattr(torch, torch_dtype)
        attn_impl = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None

        self.model = ColQwen2.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device,
            attn_implementation=attn_impl,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(pretrained_model_name_or_path)

    @property
    def use_visual_embedding(self) -> bool:
        return True

    def _process_images(self, images: List[Image.Image], **_: Any):
        return self.processor.process_images(images).to(self.model.device)

    def _process_queries(self, queries: List[str], **_: Any):
        return self.processor.process_queries(queries).to(self.model.device)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs: Any) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=_ListDataset(queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._process_queries,
        )
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Forward pass ColQwen2 queries...", leave=False):
                batch_embeddings = self.model(**batch).to("cpu")
                embeddings.extend(list(torch.unbind(batch_embeddings)))
        return embeddings

    def forward_documents(self, documents: List[Image.Image], batch_size: int, **kwargs: Any) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=_ListDataset(documents),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._process_images,
        )
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Forward pass ColQwen2 documents...", leave=False):
                batch_embeddings = self.model(**batch).to("cpu")
                embeddings.extend(list(torch.unbind(batch_embeddings)))
        return embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        document_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if hasattr(self.processor, "score_multi_vector"):
            return self.processor.score_multi_vector(query_embeddings, document_embeddings)
        return self.processor.score(query_embeddings, document_embeddings, batch_size=batch_size, device="cpu")

    def compute_metrics(self, relevant_docs: Any, results: Any, **kwargs: Any) -> Dict[str, Optional[float]]:
        try:
            from vidore_benchmark.evaluation.eval_utils import CustomRetrievalEvaluator

            evaluator = CustomRetrievalEvaluator()
        except Exception:
            from mteb.evaluation.evaluators import RetrievalEvaluator

            evaluator = RetrievalEvaluator()

        ndcg, _map, recall, precision, naucs = evaluator.evaluate(
            relevant_docs,
            results,
            evaluator.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
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
