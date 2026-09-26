# FIRE 2026 Task 1 — Statute Prediction (Team Approaching_Nirvana)

Code for our submission to **Task 1: Statute Prediction** of the FIRE 2026 shared task
*"LLM as a Judge?: From Statute Prediction to Sycophancy Detection in Law."*

Given the facts of an Indian criminal case, the system predicts the applicable Indian
Penal Code (IPC) sections, extracts the supporting sentence for each, and produces a
reasoning trace — evaluated on the expert-annotated **PROSLEX** dataset.

Our best run placed **5th of 17 teams** (Macro-F1 0.4592, Micro-F1 0.6406).

## Approach

A hybrid pipeline combining a generative predictor with a supervised classifier:

- **Qwen2.5-7B-Instruct** (4-bit NF4) — primary, open-label statute predictor. Prompted
  (zero-shot and few-shot) to emit a strict JSON object of statutes, verbatim evidence,
  and reasoning.
- **Evidence grounding** — every generated evidence span is snapped back to the closest
  verbatim sentence of the case facts, so no quoted evidence is fabricated.
- **LegalBERT** (`nlpaueb/legal-bert-base-uncased`) — fine-tuned multi-label classifier
  used as a baseline and as a **fallback** when the LLM abstains, guaranteeing at least
  one prediction per case.
- **Validation** — output is normalized, de-duplicated, and checked against the exact
  submission schema before serialization to JSONL.

See `figures/pipeline.drawio` for the architecture (open at https://app.diagrams.net).

## Repository layout

```
notebooks/Test_FIRE_task_1.ipynb   # end-to-end pipeline
figures/pipeline.drawio            # architecture diagram (editable)
requirements.txt                   # Python dependencies
```

## Running

The notebook is written for **Google Colab with a T4 GPU** (4-bit loading needs CUDA).

1. Open `notebooks/Test_FIRE_task_1.ipynb` in Colab and select a GPU runtime.
2. Place the dataset files at `/content/task1.jsonl` (train) and
   `/content/task_1_statute_prediction.jsonl` (test).
3. Run all. The pipeline writes `submission.jsonl`.

Locally: `pip install -r requirements.txt` and run the notebook with Jupyter (a CUDA GPU
is required for the 4-bit model; switch to `Qwen2.5-3B-Instruct` for smaller GPUs).

## Data

The **PROSLEX** dataset is provided by the shared-task organisers and is **not
redistributed here**. Obtain it from the task organisers / dataset release:
https://arxiv.org/abs/2608.08830

## Submission format

One JSON object per line with fields `id`, `fact`, `reasoning_traces`, and
`explanation`, where `explanation` maps each verbatim evidence sentence to a list of IPC
sections:

```json
{"id": "ST-PRED-0001", "fact": "...", "reasoning_traces": "...",
 "explanation": {"<verbatim sentence>": ["IPC 302", "IPC 34"]}}
```

## Team

**Approaching_Nirvana** — Sayanjib Sur, Chayan Ghosh, Tohida Rehman
Department of Information Technology, Jadavpur University, Kolkata, India

## License

Released under the MIT License (see `LICENSE`).
