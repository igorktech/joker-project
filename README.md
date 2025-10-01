<div align="center">

# CLEF 2025 JOKER Track: No Pun Left Behind

<a href="http://joker-project.com/" target="_blank"><img alt="JOKER Track" src="https://img.shields.io/badge/CLEF_2025-JOKER_Track-orange" height="22"></a>
<a href="https://ceur-ws.org/Vol-4038/paper_225.pdf" target="_blank" rel="noopener"><img alt="Publication (PDF)" src="https://img.shields.io/badge/Paper-CEUR--WS%204038%20(Paper%20225)-2ea44f" height="22"></a>
<a href="https://wandb.ai/igorktech01/joker-pun-translation" target="_blank"><img alt="Weights & Biases" src="https://img.shields.io/badge/Logging-wandb-6441a5" height="22"></a>
<a href="https://huggingface.co/collections/igorktech/clef-2025-joker-track-no-pun-left-behind-68bf34cc41d70a91a3709ce4" target="_blank"><img alt="Models" src="https://img.shields.io/badge/Models-HuggingFace-yellow" height="22"></a>
<a href="LICENSE" target="_blank"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue" height="22"></a>

</div>

This repository hosts an experimental codebase for the **CLEF 2025 JOKER Track** on computational wordplay. It focuses primarily on **Task 2 (Pun Translation EN→FR)** while remaining extensible to **Task 1 (Humour-aware IR)** and **Task 3 (Onomastic Wordplay Translation)**. It provides:

* Supervised fine-tuning (SFT) pipeline with LoRA
* Alternative alignment / preference optimization (ARPO-style CPO/SimPO) training
* Structured JSON config system for reproducibility
* Batched inference + submission packaging
* Optional Unsloth acceleration & 4/8-bit loading
* Integrated on-the-fly COMET (translation quality) evaluation callback (optional)

---

## 🧩 Tasks Overview

| Task | Description | Example Challenge |
|------|-------------|-------------------|
| Task 1 | Humour-aware Information Retrieval | Retrieve jokes relevant to a semantic query ("physics", "dating", etc.) preserving humorous intent. |
| Task 2 | Pun Translation (EN→FR) | Preserve dual meanings + humor: `I used to be a banker but I lost interest` → `J'ai été banquier mais j'en ai perdu tout l'intérêt`. |
| Task 3 | Onomastic Wordplay Translation | Maintain name-based wordplay (proper nouns, famous figures) while retaining pun plausibility. |

---

## ✨ Key Features

* Unified training interfaces: `src/sft.py` (supervised) and `src/arpo.py` (preference / constrained policy optimization style)
* LoRA integration (PEFT) + optional Unsloth fast adapters
* Configurable generation defaults saved alongside model artifacts
* Completion-only loss mode with response template masking
* Mid-training NMT quality probing via COMET (optional callback)
* Reproducible, declarative experiment configs (JSON)
* Submission inference helper that auto-zips predictions for Task 2

---

## 🗂 Repository Layout (Essentials)

```
src/
  sft.py                     # Supervised fine-tuning entry
  arpo.py                    # Alignment / preference optimization training
  run_submission_inference.py# Batch generation + packaging for submissions
  utils/                     # Collators, callbacks, metrics, seeding, IO
  scripts/                   # Prompt templates, helpers
configs/                     # Experiment JSON configs (SFT + ARPO)
data/                        # Place your local datasets (not tracked)
runs/                        # Shell scripts & run outputs
Experiments.ipynb            # Exploratory notebook
```

---

## 🔧 Installation

Create an environment (example with `uv` or `conda`). Dependencies are standard: `transformers`, `trl`, `datasets`, `accelerate`, `peft`, `wandb`, `unsloth` (optional), `tqdm`, `comet-ml` / `unbabel-comet` (if using COMET callback).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# (Optional) create requirements.txt later; for now install minimal stack:
pip install transformers accelerate trl peft datasets wandb tqdm unsloth
# Optional metrics (only if using NMT callback)
pip install unbabel-comet
```

Login to Hugging Face & Weights & Biases if pushing to hub / logging:

```bash
huggingface-cli login
wandb login
```

---

## 🗃 Data Preparation

Expected raw JSON lists for training / evaluation:

* SFT format (chat-style) example item:
```json
{
  "messages": [
    {"role": "user", "content": "Translate this English pun into French: I used to be a banker but I lost interest"},
    {"role": "assistant", "content": "J'ai été banquier mais j'en ai perdu tout l'intérêt"}
  ]
}
```
* (If using NMT callback) script will derive `instruction` + `target` fields during evaluation formatting when absent.

Place curated files under `data/` and reference them in a config (see below).

---

## ⚙️ Configuration Schema (Summary)

Each JSON in `configs/` fully describes a run. Core fields:

| Key | Purpose |
|-----|---------|
| `train_file` / `eval_file` | Paths to JSON lists of examples |
| `model_name` | Base HF model (chat / instruct style) |
| `lora` | PEFT LoRA block (omit or set `null` to disable) |
| `generation_config` | Saved inference defaults (temperature, beams, etc.) |
| `max_tokens_count` / `max_length` | Sequence length control |
| `completion_only` | If true, masks loss to assistant response only |
| `response_template` | String token prefix marking assistant region |
| `use_nmt_callback` | Enable COMET evaluation mid-training |
| `trainer` | Training hyperparameters passed to TRL / custom trainer |
| `seed` | Reproducibility |
| `output_dir` | Where checkpoints & tokenizer get written |

Minimal SFT config skeleton:
```json
{
  "train_file": "data/task2/train.json",
  "eval_file": "data/task2/dev.json",
  "model_name": "croissantllm/CroissantLLMChat-v0.1",
  "max_tokens_count": 512,
  "completion_only": true,
  "response_template": "<|im_start|>assistant",
  "lora": {"r": 32, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj","v_proj"]},
  "trainer": {"num_train_epochs": 1, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 4, "learning_rate": 5e-5, "eval_strategy": "steps", "eval_steps": 50, "save_steps": 200, "report_to": "wandb", "push_to_hub": true, "hub_model_id": "user/project-sft-v1"},
  "seed": 3407,
  "output_dir": "models/project-sft-v1"
}
```

---

## 🏋️ Training (Supervised Fine-Tuning)

```bash
python src/sft.py train \
  --config_file configs/skommarkhos_croissantllmchat_v0.1_1b_sft_v1.json \
  --output_dir models/sft_run_1
```

Notes:
* Set `use_unsloth=True` for faster adapter training (8-bit/4-bit)
* `completion_only` rewrites internal collator to focus on assistant spans
* Generation config is saved for downstream evaluation

---

## 🤝 Alignment / Preference Optimization (ARPO / CPO / SimPO)

ARPO training mimics constrained policy optimization with a custom trainer (`CPOTrainer`). Similar invocation:

```bash
python src/arpo.py train \
  --config_file configs/skommarkhos_croissantllmchat_v0.1_1b_arpo_v1.json \
  --output_dir models/arpo_run_1
```

Key deltas vs SFT:
* `loss_type` (e.g. `simpo`) inside `trainer`
* Separate prompt/completion length caps (`max_prompt_length`, `max_completion_length`)
* Lower learning rate typical (`5e-7` in example)

---

## 🔄 Mid-Training Evaluation (Optional NMT Callback)

Enable by setting `"use_nmt_callback": true`. The callback:
1. Derives instruction/target pairs if absent
2. Generates translations using saved `generation_config`
3. Scores with COMET22 (if `unbabel-comet` installed)
4. Logs metrics (W&B if enabled)

Runs ~2 times per training by dynamically spacing evaluation steps.

---

## 🚀 Inference & Submission Packaging (Task 2)

```bash
python src/run_submission_inference.py main \
  --model_path models/sft_run_1 \
  --test_data data/task2/joker_pun_translation_2025_test.json \
  --output_dir submissions/task2 \
  --batch_size 32
```

Outputs:
* JSON with fields: `run_id`, `manual`, `id_en`, `en`, `fr`
* Auto-generated ZIP containing `prediction.json` (ready for upload)

Temperature / sampling settings currently defined inline (tune as desired inside `run_submission_inference.py`).

---

## 🧪 Reproducibility Checklist

* Fixed seed in config (`seed`)
* Explicit tokenizer + special tokens saved to `output_dir`
* Generation parameters versioned
* LoRA adapter weights merged only if you export them explicitly (default: PEFT format)

---

## 🛠 Extending

| Goal | Where to Modify |
|------|-----------------|
| New metric | `src/utils/metrics.py` |
| Alternate reward / loss | `utils/cpo_trainer.py` (custom trainer) |
| Prompt template logic | `src/scripts/prompt.py` |
| Custom collator | `utils/collators.py` |

---

## 📌 Roadmap (Planned)

* Add retrieval baseline for Task 1 (BM25 + reranker)
* Add name-entity augmentation patterns for Task 3
* Publish structured requirements file & lightweight Dockerfile
* Add evaluation harness for BLEU / chrF / pun-preservation score
* Merge LoRA weights export utility script

---

## 📄 License

This project is licensed under the Apache License 2.0.

```
Copyright 2025 Igor Kuzmin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## ✍️ Citation

If you use this codebase or derivatives in academic work, please cite:

```
@inproceedings{kuzmin2025joker,
  author    = {Igor Kuzmin},
  title     = {{CLEF} 2025 {JOKER} Track: No Pun Left Behind},
  booktitle = {CLEF 2025 Labs and Workshops, Notebook Papers},
  series    = {CEUR Workshop Proceedings},
  volume    = {4038},
  publisher = {CEUR-WS.org},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4038/paper_225.pdf},
  issn      = {1613-0073},
  note      = {Paper 225}
}
```
