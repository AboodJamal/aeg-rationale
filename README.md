# Instruction-Tuning Mistral-7B for Automated Essay Scoring & Rationale Generation

> Efficient fine-tuning of **Mistral-7B-Instruct** (4-bit, Unsloth) with **LoRA** for **Automated Essay Grading (AEG)** and **explainable rationales**.

---

## 🧭 Project Overview

This project instruction-tunes `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` to (1) grade essays and (2) generate human-like rationales that justify each score. It combines **4-bit quantization** for memory efficiency with **PEFT/LoRA** to adapt the model to AEG while training only a small fraction of parameters. The training data is a **3552-sample** custom set built via **knowledge distillation** from a **GPT-4o mini** teacher, and covers diverse academic and general topics. fileciteturn0file0

**Why this matters:** Traditional AEG systems struggled with semantic nuance and explainability. An instruction-tuned LLM can follow rubrics, align with reference answers, and articulate *why* a score was assigned—bringing grading quality closer to expert assessors. fileciteturn0file0

---

## ✨ Key Features

- **Task:** Automated Essay Scoring + Rationale Generation from question, reference answer, student response, and mark scheme. fileciteturn0file0  
- **Model:** `Mistral-7B-Instruct` (Unsloth, **bnb-4bit**) with **LoRA** adapters (Q/K/V/O/FFN projections). fileciteturn0file0  
- **Data:** 3,552 instruction-format samples (Alpaca style: `instruction`, `input`, `output`). fileciteturn0file0  
- **Metrics:** Exact Match, MAE, RMSE, Cohen’s Kappa, Weighted F1, Pearson & Spearman correlations, tolerance-1 accuracy. fileciteturn0file0  
- **Infra:** SFTTrainer, gradient checkpointing, mixed precision; **W&B** for experiment tracking; **Gradio** demo UI. fileciteturn0file0

---

## 📚 Dataset

- **Size:** 3,552 samples distilled from **GPT-4o mini** to transfer grading & rationale style to Mistral-7B. fileciteturn0file0  
- **Coverage:** NLP, CS/SE, Big Data, DB, DSA, IR, ML, plus chemistry, football, martial arts, medicine, Palestinian history, programming, and general knowledge—improves robustness and generalization. fileciteturn0file0  
- **Format:**  
  - `instruction`: fixed prompt — *“Grade the student's answer… Give a score and rationale.”*  
  - `input`: multi-line context block with **Question**, **Reference Answer**, **Student Answer**, **Mark Scheme**.  
  - `output`: **Score** (numeric) + **Rationale** (text). fileciteturn0file0

**Splits:** Train 80% (2,842), Val 10% (355), Test 10% (355). fileciteturn0file0

---

## 🧪 Methodology

1. **Base model & quantization**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` for memory/perf efficiency. fileciteturn0file0  
2. **PEFT/LoRA**: adapters on attention and MLP projections; fast convergence, low memory, preserves base knowledge. fileciteturn0file0  
3. **Alpaca-style tuning**: consistent instruction templates; careful sequence/tokenization management. fileciteturn0file0  
4. **Hyperparameter search**: LR ∈ {1e-5 … 3e-4}; LoRA **r** ∈ {16,32,64,128}, **α** ∈ {16,32,64,128}, dropout ∈ {0.0,0.1,0.2}; batch sizes {2,4,8} (with accumulation); schedulers {linear, cosine, polynomial}; warmup {5%,10%,15%}. fileciteturn0file0  
5. **Iterative improvements**: dataset rebalancing; curriculum learning from easy→hard; prompt-template experiments (incl. structured outputs, explicit criteria). fileciteturn0file0  
6. **Evaluation**: quantitative (accuracy, error, correlation, agreement, per-class F1) + qualitative (rationale quality, consistency, bias checks, edge cases). fileciteturn0file0

---

## 📈 Results (Best LoRA: r=32, α=32)

| Metric | r=8, α=8 | r=16, α=16 | **r=32, α=32 (Best)** |
|---|---:|---:|---:|
| Exact Match Accuracy | 0.7893 | 0.7949 | **0.8258** |
| Tolerance-1 Accuracy (±1 pt) | 0.9972 | 0.9972 | 0.9972 |
| MAE | 0.2163 | 0.2107 | **0.1798** |
| RMSE | 0.4829 | 0.4770 | **0.4434** |
| Score Bias (pred−true) | 0.0084 | 0.0084 | 0.0225 |
| Cohen’s Kappa | 0.7182 | 0.7259 | **0.7660** |
| Weighted F1 | 0.7884 | 0.7934 | **0.8241** |
| Pearson r | 0.9282 | 0.9300 | **0.9399** |
| Spearman ρ | 0.9221 | 0.9234 | **0.9333** |
| Correct / Over / Under | 281/40/35 | 283/39/34 | **294/36/26** |

**Takeaways:** Larger LoRA capacity improved exact-match accuracy (~+4.6% vs r=8), reduced MAE/RMSE, and raised Kappa to **0.766** (substantial agreement), with fewer over/under-grades—indicating fairer, more consistent scoring. fileciteturn0file0

---

## 🛠️ Setup

> Requires Python 3.10+ and a CUDA-capable GPU (recommended).

```bash
# clone your repo
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

# create env
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# install core deps
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA
pip install transformers accelerate datasets peft bitsandbytes trl evaluate
pip install unsloth wandb gradio scikit-learn numpy pandas matplotlib
```

If using **W&B**:
```bash
wandb login
```

---

## 🚀 Training

Minimal example (adapt to your file paths and script names):

```bash
python train_sft.py   --model_name unsloth/mistral-7b-instruct-v0.3-bnb-4bit   --dataset_path data/aeg_alpaca.jsonl   --output_dir outputs/mistral-aeg-lora   --lora_r 32   --lora_alpha 32   --lora_dropout 0.1   --per_device_train_batch_size 2   --gradient_accumulation_steps 8   --learning_rate 2e-4   --lr_scheduler_type cosine   --warmup_ratio 0.1   --num_train_epochs 3   --bf16 True   --report_to wandb
```

> The project reported experiments across LR, LoRA rank/alpha/dropout, schedulers, and warmup ratios; best results came from **r=32, α=32**. fileciteturn0file0

---

## 🧾 Inference (Scoring + Rationale)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "outputs/mistral-aeg-lora"  # or your HF hub repo
tok = AutoTokenizer.from_pretrained(name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

instruction = "Grade the student's answer to the essay question based on the reference answer and the provided mark scheme. Give a score and rationale."
input_block = """
Question: ...
Reference Answer: ...
Student Answer: ...
Mark Scheme: ...
"""

prompt = f"Instruction:\n{instruction}\n\nInput:\n{input_block}\n\nOutput:"
ids = tok(prompt, return_tensors="pt").to(model.device)
gen = model.generate(**ids, max_new_tokens=512)
print(tok.decode(gen[0], skip_special_tokens=True))
```

---

## 🧪 Evaluation

We report (at minimum):

- **Accuracy:** exact match; tolerance-1 (±1 pt).  
- **Errors:** MAE, MSE, RMSE.  
- **Agreement:** **Cohen’s Kappa** (chance-corrected).  
- **Balance:** Weighted **F1**; per-score precision/recall/F1.  
- **Correlation:** Pearson (linear) & Spearman (rank). fileciteturn0file0

Stat tests: paired t-tests, effect sizes (Cohen’s *d*), CIs; multiple-comparison correction across hyperparam sweeps. fileciteturn0file0

---

## 🖥️ Gradio Demo

A lightweight UI was built for **real-time grading** with bundled examples + custom input. See `app.py` and launch:

```bash
python app.py
```

This renders a textbox for the four inputs and returns **score + rationale**. fileciteturn0file0

---

## 📊 Experiment Tracking

We use **Weights & Biases (W&B)** to log configs, loss curves, validation metrics, and store artifacts; experiments are grouped and tagged for comparison dashboards. fileciteturn0file0

---

## 🗂️ Suggested Repo Structure

```
.
├── data/
│   └── aeg_alpaca.jsonl
├── src/
│   ├── train_sft.py
│   ├── data_utils.py
│   ├── eval.py
│   └── modeling/
├── app.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🛣️ Roadmap

- Improve rationale structure & rubric-grounding (template-guided decoding). fileciteturn0file0  
- Expand domains and balance long-tail scores. fileciteturn0file0  
- Robust bias & fairness checks; adversarial/edge-case stress-tests. fileciteturn0file0  
- Integrate into LMS pipelines; batch & asynchronous grading. fileciteturn0file0

---

## 👥 Authors

Amro Eid · Hossam Shehadeh · Abdullah Jamal (Computer Science Apprenticeship Program, An-Najah National University). Contact emails in the paper. fileciteturn0file0

---

## 📄 Citation

If you use this repository or build on our methods, please cite the project paper:

> **Instruction-Tuning Mistral-7B-Instruct-v0.3 for Automated Essay Scoring and Rationale Generation**.  
> Amro Eid, Hossam Shehadeh, Abdullah Jamal. 2025.

---

## 🛡️ License

Choose a license (e.g., Apache-2.0, MIT).

---

## 🔗 References

- Mistral-7B-Instruct (HF)  
- Unsloth docs & 4-bit guides  
- PEFT/LoRA papers & examples  
- SFT/TRL trainer docs  
(See the paper’s reference list.) fileciteturn0file0
