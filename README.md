# MMhops-R1: Multimodal Multi-hop Reasoning

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2512.13573)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/taoszhang/MMhops)
[![Knowledge Base](https://img.shields.io/badge/Knowledge_Base-HuggingFace-orange)](https://huggingface.co/datasets/taoszhang/MMhops-KB)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## 📖 Introduction

This repository contains the dataset **MMhops** and the official implementation of **MMhops-R1**, as proposed in the paper: **"MMhops-R1: Multimodal Multi-hop Reasoning"** (AAAI 2026).

Multimodal Large Language Models (MLLMs) have made significant progress but are often limited to single-step reasoning. Real-world problems, however, frequently require iteratively integrating information across various modalities and external knowledge.

To bridge this gap, we introduce:
1.  **MMhops**: A novel, large-scale benchmark designed to systematically evaluate and foster multimodal multi-hop reasoning.
2.  **MMhops-R1**: A Multi-modal Retrieval-Augmented Generation (mRAG) framework that uses reinforcement learning to autonomously plan reasoning paths and query external knowledge.

## 📚 MMhops Dataset

**MMhops** is the first large-scale dataset focused on complex reasoning chains that span both visual and textual modalities. It comprises **31,117 samples** derived from Wikipedia, covering **20,483 unique questions** and **28,256 images**.

### 🔗 Download
You can download the dataset directly from Hugging Face:
> **[Hugging Face Dataset: taoszhang/MMhops](https://huggingface.co/datasets/taoszhang/MMhops)**

### 🧩 Dataset Construction & Tasks
MMhops includes two challenging task formats:

1.  **Bridging Reasoning**: Starts from a single image and requires multi-step chain reasoning (2+ hops) to link visual entities to external textual knowledge.
2.  **Comparison Reasoning**: Involves identifying entities across multiple images and performing comparative analysis based on their shared attributes.

<img width="1234" height="533" alt="fig2" src="https://github.com/user-attachments/assets/0e3683ac-fbe0-4c15-895b-b4ebd852e293" />
The multi-stage construction process for the MMhops dataset. (a) Bridging Dataset Construction iteratively expands questions to increase hop depth. (b) Comparison Dataset Construction generates questions comparing attributes of entities from different images.

### 📊 Statistics
Unlike existing KB-VQA datasets (e.g., OK-VQA, InfoSeek) which are often limited to single-hop reasoning, MMhops significantly increases complexity:
* **100%** of samples require external knowledge.
* **70.8%** of samples require 3 reasoning steps.
* **29.2%** of samples require 4 reasoning steps.

### 📋 Dataset Schema

Each sample in the dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier (e.g., `"MMhops_test_00001"`) |
| `problem` | string | Question text with `<image>` placeholder |
| `answer` | string | Reference answer |
| `answer_eval` | list[string] | Ground truth values for evaluation |
| `split` | string | Task type: `"Bridge"` or `"Compare"` |
| `problem_type` | string | Answer type: `"String"`, `"Numerical"`, or `"Time"` |

**Test Set Distribution:**

| Split | Count | Question Type | Count |
|-------|-------|---------------|-------|
| Bridge | 5,287 | Numerical | 3,954 |
| Compare | 936 | String | 1,300 |
| **Total** | **6,223** | Time | 969 |

## 🤖 MMhops-R1 Framework

<img width="1487" height="405" alt="fig3" src="https://github.com/user-attachments/assets/aa17d14e-39ff-4e82-a1cd-94705c1fbe57" />
**MMhops-R1** is an RL-driven mRAG framework designed for dynamic reasoning. Unlike static retrieval pipelines, MMhops-R1 can autonomously select reasoning strategies and interact with retrievers.

### Key Features
* **Action Space**: The model supports three core actions:
    1.  `Image Retrieval` ($a_{is}$): Selects an input image to query the image retriever.
    2.  `Text Retrieval` ($a_{ts}$): Submits a text query to the text retriever.
    3.  `Answer` ($a_{a}$): Generates the final response based on gathered information.
* **Reward Modeling**: We employ a composite reward function $R(\tau)$:
    * $R_{outcome}$: Correctness of the final answer.
    * $R_{format}$: Adherence to structured tags (e.g., `<text_search>`).
    * $R_{action}$: Incentivizes effective tool use, gated by success.

## 🏆 Main Results

We evaluated MMhops-R1 against strong baselines, including proprietary MLLMs and existing Multi-hop/Multimodal RAG methods. MMhops-R1 achieves state-of-the-art performance among open-source methods.

| Method | Base Model | Retriever | Bridging | Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-4o** | - | - | 36.62 | 8.76 |
| **Gemini-2.5-Pro** | - | - | 53.98 | 29.39 |
| **Search-r1** | Qwen2.5-7B | Caption, Text | 19.98 | 6.62 |
| **OmniSearch** | GPT-4o | Image, Text | 42.65 | 17.02 |
| **MMhops-R1 (Ours)**| **Qwen2.5-VL-7B** | **Image, Text** | **51.35** | **22.01** |

*Table 1: Main results on the MMhops test set.*

## 🔎 Retrieval Services

The prebuilt text and image retrieval databases are available from
**[Hugging Face: taoszhang/MMhops-KB](https://huggingface.co/datasets/taoszhang/MMhops-KB)**.
They contain 904,790 E5-indexed Wikipedia passages and 96,110 CLIP-indexed
Wikipedia entities. Raw images and encoder weights remain external dependencies.

This repository provides two self-hosted FastAPI services:

- `retrieval/text_retrieval_server.py`: E5 text retrieval at `POST /retrieve`.
- `retrieval/image_retrieval_server.py`: CLIP image retrieval at `POST /image_search`.

Download the knowledge base and install the service dependencies with:

```bash
pip install -r retrieval/requirements.txt
hf download taoszhang/MMhops-KB \
  --repo-type dataset --local-dir data/MMhops-KB
```

See **[retrieval/README.md](retrieval/README.md)** for launch commands, request
examples, index configuration, and deployment notes.

## 📐 Evaluation

We provide an official evaluation script to benchmark your model on MMhops.

### Installation

```bash
cd evaluation
pip install -r requirements.txt
```

### Prediction File Format

Your predictions should be saved in **JSONL format** (one JSON object per line):

```jsonl
{"id": "MMhops_test_00001", "prediction": "174th"}
{"id": "MMhops_test_00002", "prediction": "12"}
{"id": "MMhops_test_00003", "prediction": "Eight"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Sample ID matching the dataset |
| `prediction` | string | ✅ | Model's predicted answer |

### Run Evaluation

```bash
python evaluation/evaluate_mmhops.py --prediction-file your_predictions.jsonl
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--prediction-file`, `-p` | (required) | Path to prediction file (JSONL) |
| `--dataset` | `taoszhang/MMhops` | HuggingFace dataset name |
| `--split` | `test` | Dataset split: `test`, `train`, or `validation` |
| `--output-json`, `-o` | None | Save results to JSON file |

### Evaluation Metrics

| Question Type | Evaluation Method |
|---------------|-------------------|
| **String** | Exact match after normalization (lowercase, remove articles/punctuation) |
| **Numerical** | Range matching: correct if prediction falls within `[min, max]` or IoU ≥ 0.5 |
| **Time** | Exact match after normalization |

## 🖊️ Citation

If you find this dataset or code useful in your research, please cite our paper:

```bibtex
@misc{zhang2025mmhopsr1multimodalmultihopreasoning,
      title={MMhops-R1: Multimodal Multi-hop Reasoning}, 
      author={Tao Zhang and Ziqi Zhang and Zongyang Ma and Yuxin Chen and Bing Li and Chunfeng Yuan and Guangting Wang and Fengyun Rao and Ying Shan and Weiming Hu},
      year={2025},
      eprint={2512.13573},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13573}, 
}
```
