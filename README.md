# MMhops-R1: Multimodal Multi-hop Reasoning

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2512.13573) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/taoszhang/MMhops)
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

| Method | Base Model | Retriever | Bridging (Overall) | Comparison |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-4o** | - | - | 36.62 | 8.76 |
| **Gemini-2.5-Pro** | - | - | 53.98 | 29.39 |
| **Search-r1** | Qwen2.5-7B | Caption, Text | 19.98 | 6.62 |
| **OmniSearch** | GPT-4o | Image, Text | 42.65 | 17.02 |
| **MMhops-R1 (Ours)**| **Qwen2.5-VL-7B** | **Image, Text** | **51.35** | **22.01** |

*Table 1: Main results on the MMhops test set.*

## 🖊️ Citation

If you find this dataset or code useful in your research, please cite our paper:

@misc{zhang2025mmhopsr1multimodalmultihopreasoning,
      title={MMhops-R1: Multimodal Multi-hop Reasoning}, 
      author={Tao Zhang and Ziqi Zhang and Zongyang Ma and Yuxin Chen and Bing Li and Chunfeng Yuan and Guangting Wang and Fengyun Rao and Ying Shan and Weiming Hu},
      year={2025},
      eprint={2512.13573},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13573}, 
}
