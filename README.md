<div align="center">

# ⚖️ Grounded Statute Prediction — FIRE 2026 Task 1

### Team *Approaching Nirvana*

Hybrid **LLM + LegalBERT** pipeline for the FIRE 2026 shared task
*"LLM as a Judge?: From Statute Prediction to Sycophancy Detection in Law."*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E)](https://github.com/huggingface/transformers)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--7B--Instruct-6f42c1)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Colab](https://img.shields.io/badge/Runs%20on-Colab%20T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Rank](https://img.shields.io/badge/Official%20Rank-5th%20%2F%2017-gold)

</div>

---

## 🎯 The Task

Given the **facts** of an Indian criminal case, the system must:

1. 📜 predict the applicable **Indian Penal Code (IPC)** sections,
2. 🔎 point to the **verbatim sentence** that supports each one, and
3. 🧠 produce a **reasoning trace** explaining the prediction.

Evaluated on the expert-annotated **PROSLEX** dataset.

## 🏆 Results

Our best run (**Run 1**) placed **5th of 17 teams**.

| Macro-F1 | Micro-F1 | Accuracy | ROUGE-L | BLEU |
|:--------:|:--------:|:--------:|:-------:|:----:|
| 0.4592   | 0.6406   | 0.4736   | 0.1660  | 0.0817 |

## 🧩 Approach

A hybrid pipeline that keeps a generative model's flexibility while enforcing evidential fidelity.

| Stage | Component | Role |
|------|-----------|------|
| 🟢 Predict | **Qwen2.5-7B-Instruct** (4-bit NF4) | Open-label statute prediction via a strict JSON prompt (zero- & few-shot) |
| 🟢 Ground | **difflib snapping** | Every evidence span is snapped to a verbatim fact sentence — no fabricated quotes |
| 🟠 Fallback | **LegalBERT** (fine-tuned) | Supervised baseline **and** fallback so every case gets ≥ 1 statute |
| 🔵 Validate | **schema check** | Normalize, de-duplicate, verify verbatim keys + schema before writing JSONL |

<div align="center">

*Architecture diagram: [`figures/pipeline.drawio`](figures/pipeline.drawio) — open at [diagrams.net](https://app.diagrams.net).*

</div>

## 📂 Repository Layout

```
📦 fire2026-task1-statute-prediction
├── 📓 notebooks/Test_FIRE_task_1.ipynb   # end-to-end pipeline
├── 🖼️ figures/pipeline.drawio            # architecture diagram (editable)
├── 📋 requirements.txt                   # Python dependencies
└── 📄 LICENSE
```

## 🚀 Running

Written for **Google Colab with a T4 GPU** (4-bit loading needs CUDA).

```bash
# local install (a CUDA GPU is required for 4-bit; use Qwen2.5-3B-Instruct on smaller GPUs)
pip install -r requirements.txt
```

1. Open [`notebooks/Test_FIRE_task_1.ipynb`](notebooks/Test_FIRE_task_1.ipynb) in Colab, select a **GPU** runtime.
2. Place the dataset at `/content/task1.jsonl` (train) and `/content/task_1_statute_prediction.jsonl` (test).
3. **Run all** → the pipeline writes `submission.jsonl`.

## 🗂️ Data

The **PROSLEX** dataset belongs to the shared-task organisers and is **not redistributed here**.
Obtain it from the official release → 🔗 https://arxiv.org/abs/2608.08830

## 🧾 Submission Format

One JSON object per line — `explanation` maps each verbatim evidence sentence to a list of IPC sections:

```json
{
  "id": "ST-PRED-0001",
  "fact": "...",
  "reasoning_traces": "...",
  "explanation": { "<verbatim sentence>": ["IPC 302", "IPC 34"] }
}
```

## 👥 Team

**Approaching Nirvana** — Department of Information Technology, Jadavpur University, Kolkata, India

- Sayanjib Sur
- Chayan Ghosh

*Under the guidance of **Dr. Tohida Rehman**.*

## 📜 License

Released under the [MIT License](LICENSE).
