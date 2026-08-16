#!/usr/bin/env python3
"""Serve the MMhops CLIP/FAISS image retriever through FastAPI."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
from pathlib import Path
from typing import Any, Optional, Union

import clip
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


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


def load_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            record = json.loads(line)
            if int(record["id"]) != row:
                raise ValueError(
                    f"Corpus ID mismatch at row {row}: found {record['id']!r}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Corpus is empty: {path}")
    return records


class ImageRetriever:
    def __init__(
        self,
        index_path: Path,
        corpus_path: Path,
        model_name: str,
        device_name: str,
        batch_size: int,
        use_faiss_gpu: bool,
        allow_local_paths: bool,
        image_root: Optional[Path],
        max_image_bytes: int,
    ) -> None:
        self.device = select_device(device_name)
        self.corpus = load_corpus(corpus_path)
        self.index = faiss.read_index(str(index_path.resolve(strict=True)))
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Image index must use inner-product search")
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
        self.model, self.preprocess = clip.load(
            model_name, device=self.device, jit=False
        )
        self.model.eval()
        model_dimension = getattr(self.model.visual, "output_dim", None)
        if model_dimension is not None and model_dimension != self.index.d:
            raise ValueError(
                f"Model/index dimension mismatch: {model_dimension} != {self.index.d}"
            )
        self.batch_size = batch_size
        self.allow_local_paths = allow_local_paths
        self.image_root = image_root.resolve(strict=True) if image_root else None
        self.max_image_bytes = max_image_bytes

    def open_local_image(self, value: str) -> Image.Image:
        if not self.allow_local_paths:
            raise ValueError(
                "Local image paths are disabled; send image_base64 or start the "
                "server with --allow-local-paths"
            )
        path = Path(value).expanduser().resolve(strict=True)
        if self.image_root is not None:
            try:
                path.relative_to(self.image_root)
            except ValueError as error:
                raise ValueError(
                    f"Image path is outside --image-root: {path}"
                ) from error
        if path.stat().st_size > self.max_image_bytes:
            raise ValueError(f"Image exceeds {self.max_image_bytes} bytes: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    def open_base64_image(self, value: str) -> Image.Image:
        payload = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("image_base64 contains invalid base64 data") from error
        if len(raw) > self.max_image_bytes:
            raise ValueError(f"Decoded image exceeds {self.max_image_bytes} bytes")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        except UnidentifiedImageError as error:
            raise ValueError("Decoded payload is not a supported image") from error

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            tensor = torch.stack(
                [self.preprocess(image) for image in images[start : start + self.batch_size]]
            ).to(self.device)
            features = self.model.encode_image(tensor)
            features = torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

    def search(self, images: list[Image.Image], topk: int) -> list[list[dict[str, Any]]]:
        if not images:
            raise ValueError("At least one image is required")
        if topk < 1 or topk > min(100, self.index.ntotal):
            raise ValueError("topk must be between 1 and 100")
        scores, indices = self.index.search(self.encode(images), topk)
        results: list[list[dict[str, Any]]] = []
        for row_indices, row_scores in zip(indices, scores):
            hits: list[dict[str, Any]] = []
            for index, score in zip(row_indices.tolist(), row_scores.tolist()):
                hit = dict(self.corpus[index])
                hit["score"] = float(score)
                hits.append(hit)
            results.append(hits)
        return results


ImageInput = Optional[Union[str, list[str]]]


class ImageSearchRequest(BaseModel):
    image_paths: ImageInput = None
    image_base64: ImageInput = None
    topk: Optional[int] = None


app = FastAPI(title="MMhops image retrieval service")
retriever: Optional[ImageRetriever] = None
default_topk = 5


def as_list(value: Union[str, list[str]]) -> tuple[list[str], bool]:
    if isinstance(value, str):
        return [value], True
    return value, False


@app.get("/health")
def health() -> dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not loaded")
    return {
        "status": "ok",
        "model": retriever.model_name,
        "dimension": retriever.index.d,
        "entries": retriever.index.ntotal,
        "index_features": "0.5 * normalized image + 0.5 * normalized title",
    }


@app.post("/image_search")
def image_search(request: ImageSearchRequest) -> dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not loaded")
    if (request.image_paths is None) == (request.image_base64 is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of image_paths or image_base64",
        )

    try:
        if request.image_paths is not None:
            values, single = as_list(request.image_paths)
            images = [retriever.open_local_image(value) for value in values]
        else:
            assert request.image_base64 is not None
            values, single = as_list(request.image_base64)
            images = [retriever.open_base64_image(value) for value in values]
        topk = request.topk if request.topk is not None else default_topk
        results = retriever.search(images, topk)
    except (OSError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    legacy_results = [
        [[hit["title"], hit["score"]] for hit in image_results]
        for image_results in results
    ]
    if single:
        return {
            "mode": "single",
            "results": legacy_results[0],
            "records": results[0],
        }
    return {"mode": "batch", "results": legacy_results, "records": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--model", default="ViT-L/14@336px")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--faiss-gpu", action="store_true")
    parser.add_argument(
        "--allow-local-paths",
        action="store_true",
        help="Allow clients to submit image paths visible to the server",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Restrict submitted local paths to this directory",
    )
    parser.add_argument("--max-image-bytes", type=int, default=20_000_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    return parser.parse_args()


def main() -> None:
    global default_topk, retriever
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.topk < 1 or args.topk > 100:
        raise ValueError("--topk must be between 1 and 100")
    if args.max_image_bytes < 1:
        raise ValueError("--max-image-bytes must be positive")
    if args.image_root and not args.allow_local_paths:
        raise ValueError("--image-root requires --allow-local-paths")
    default_topk = args.topk
    retriever = ImageRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_name=args.model,
        device_name=args.device,
        batch_size=args.batch_size,
        use_faiss_gpu=args.faiss_gpu,
        allow_local_paths=args.allow_local_paths,
        image_root=args.image_root,
        max_image_bytes=args.max_image_bytes,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
