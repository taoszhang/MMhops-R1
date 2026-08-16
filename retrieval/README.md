# MMhops Retrieval Services

These two FastAPI services reproduce the E5 text retrieval and CLIP image
retrieval used by MMhops-R1. The prebuilt corpora and row-aligned FAISS indexes
are published in [`taoszhang/MMhops-KB`](https://huggingface.co/datasets/taoszhang/MMhops-KB).

## Setup

```bash
pip install -r retrieval/requirements.txt
hf download taoszhang/MMhops-KB \
  --repo-type dataset --local-dir data/MMhops-KB
```

`faiss-cpu` is sufficient for serving either index. For GPU FAISS search,
install a CUDA-compatible FAISS build and add `--faiss-gpu` to the commands
below. The Faiss project officially distributes CPU and GPU builds through
Conda; `requirements.txt` uses the commonly available CPU-only PyPI wheel for
convenience. Encoder inference automatically uses CUDA when it is available.

## Text Retrieval

```bash
python retrieval/text_retrieval_server.py \
  --index-path data/MMhops-KB/text/e5_Flat.index \
  --corpus-path data/MMhops-KB/text/text_corpus.jsonl \
  --fp16
```

The service listens on `127.0.0.1:8000` by default. Submit batched queries to
the experiment-compatible `/retrieve` endpoint:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who wrote The Old Man and the Sea?"],"topk":3,"return_scores":true}'
```

The server adds the E5 `query:` prefix, applies mean pooling and L2
normalization, and returns `id`/`contents` records from `text_corpus.jsonl`.

## Image Retrieval

For trusted local use, enable server-visible image paths:

```bash
python retrieval/image_retrieval_server.py \
  --index-path data/MMhops-KB/image/CLIP_Flat.index \
  --corpus-path data/MMhops-KB/image/image_corpus.jsonl \
  --allow-local-paths
```

```bash
curl -X POST http://127.0.0.1:9999/image_search \
  -H 'Content-Type: application/json' \
  -d '{"image_paths":"/absolute/path/to/query.jpg","topk":5}'
```

Remote clients should send a base64-encoded image through `image_base64`
instead of exposing server file paths. Both fields also accept lists for batch
search. Local paths are disabled by default; use `--image-root` with
`--allow-local-paths` when the service is exposed beyond a trusted machine.
The corresponding JSON body is
`{"image_base64":"<base64 data or data URL>","topk":5}`.

The query is a normalized OpenAI CLIP `ViT-L/14@336px` image feature. Each
indexed vector is `0.5 * normalized_image + 0.5 * normalized_title`, matching
the released index construction. The `results` field retains the original
`[title, score]` API format used to prepare MMhops, while `records` returns
`id`, `title`, `entity_id`, `image_path`, and `score` from `image_corpus.jsonl`.

The database images themselves are not required to serve the prebuilt index;
only incoming query images are encoded online. Loading the CPU indexes requires
approximately 2.8 GB for text and 0.3 GB for images, excluding model memory.

## Health Checks and Network Access

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:9999/health
```

Use `--host 0.0.0.0` only when remote access is required, and place
authentication, TLS, request limits, and network filtering in front of any
public deployment. Interactive API documentation is available at `/docs`.
