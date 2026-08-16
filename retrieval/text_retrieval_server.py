#!/usr/bin/env python3
"""Serve the MMhops E5/FAISS text retriever through FastAPI."""

from __future__ import annotations

import argparse
import json
import mmap
from array import array
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class JsonlCorpus:
    """Random-access JSONL reader whose row number is the FAISS vector ID."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        self._handle = self.path.open("rb")
        if self.path.stat().st_size == 0:
            raise ValueError(f"Corpus is empty: {self.path}")
        self._offsets = array("Q", [0])
        cursor = 0
        for line in self._handle:
            cursor += len(line)
            self._offsets.append(cursor)
        if len(self._offsets) < 2:
            raise ValueError(f"Corpus has no records: {self.path}")
        self._handle.seek(0)
        self._mapping = mmap.mmap(
            self._handle.fileno(), length=0, access=mmap.ACCESS_READ
        )

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, end = self._offsets[index], self._offsets[index + 1]
        record = json.loads(self._mapping[start:end].rstrip(b"\r\n"))
        if int(record["id"]) != index:
            raise ValueError(
                f"Corpus ID mismatch at row {index}: found {record['id']!r}"
            )
        return record


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_index_to_gpu(index: Any, device: torch.device) -> tuple[Any, Any]:
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("--faiss-gpu requires a GPU-enabled FAISS installation")
    gpu_id = device.index or 0
    resources = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(resources, gpu_id, index), resources


class TextRetriever:
    def __init__(
        self,
        index_path: Path,
        corpus_path: Path,
        model_name: str,
        device_name: str,
        batch_size: int,
        max_length: int,
        use_fp16: bool,
        use_faiss_gpu: bool,
    ) -> None:
        self.device = select_device(device_name)
        self.corpus = JsonlCorpus(corpus_path)
        self.index = faiss.read_index(str(index_path.resolve(strict=True)))
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Text index must use inner-product search")
        self._faiss_resources = None
        if use_faiss_gpu:
            if self.device.type != "cuda":
                raise RuntimeError("--faiss-gpu requires a CUDA device")
            self.index, self._faiss_resources = move_index_to_gpu(
                self.index, self.device
            )

        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                "Index/corpus size mismatch: "
                f"{self.index.ntotal:,} vectors != {len(self.corpus):,} rows"
            )

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        model_dimension = getattr(self.model.config, "hidden_size", None)
        if model_dimension is not None and model_dimension != self.index.d:
            raise ValueError(
                f"Model/index dimension mismatch: {model_dimension} != {self.index.d}"
            )
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        if self.use_fp16:
            self.model.half()
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.inference_mode()
    def encode(self, queries: list[str]) -> np.ndarray:
        prefixed = [f"query: {query}" for query in queries]
        inputs = self.tokenizer(
            prefixed,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        mask = inputs["attention_mask"][..., None].bool()
        hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
        embeddings = hidden.sum(dim=1) / mask.sum(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def search(
        self, queries: list[str], topk: int, return_scores: bool
    ) -> list[list[dict[str, Any]]]:
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("queries must contain at least one non-empty string")
        if topk < 1 or topk > min(100, self.index.ntotal):
            raise ValueError("topk must be between 1 and 100")

        all_results: list[list[dict[str, Any]]] = []
        for start in range(0, len(queries), self.batch_size):
            embeddings = self.encode(queries[start : start + self.batch_size])
            scores, indices = self.index.search(embeddings, topk)
            for row_indices, row_scores in zip(indices, scores):
                hits: list[dict[str, Any]] = []
                for index, score in zip(row_indices.tolist(), row_scores.tolist()):
                    document = self.corpus[index]
                    if return_scores:
                        hits.append({"document": document, "score": float(score)})
                    else:
                        hits.append(document)
                all_results.append(hits)
        return all_results


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI(title="MMhops text retrieval service")
retriever: Optional[TextRetriever] = None
default_topk = 3


@app.get("/health")
def health() -> dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not loaded")
    return {
        "status": "ok",
        "model": retriever.model_name,
        "dimension": retriever.index.d,
        "entries": retriever.index.ntotal,
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not loaded")
    try:
        results = retriever.search(
            request.queries,
            request.topk if request.topk is not None else default_topk,
            request.return_scores,
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"result": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    global default_topk, retriever
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.topk < 1 or args.topk > 100:
        raise ValueError("--topk must be between 1 and 100")
    default_topk = args.topk
    retriever = TextRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_name=args.model,
        device_name=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=args.fp16,
        use_faiss_gpu=args.faiss_gpu,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
