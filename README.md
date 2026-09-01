# 🍽️ Qwen2.5-VL Food Attribute & Ingredient Extraction (LoRA)

An end-to-end Vision-Language Model (VLM) fine-tuning and evaluation pipeline built on **Qwen2.5-VL-3B-Instruct** using **LoRA (Low-Rank Adaptation)** to extract structured JSON metadata (food verification, title, ingredients, and drinks) directly from raw food images.

---

## 📌 Project Motivation & Objectives

Extracting unstructured information from images into structured formats is a fundamental problem in multimodal AI. Traditional object detection and classification models struggle with hierarchical attributes and fine-grained ingredient discovery. 

This project solves this by adapting a modern **Vision-Language Model (VLM)** to act as an end-to-end visual parser that maps food photography directly into a strictly validated JSON schema.

### Key Highlights
- **Base Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Dataset:** `mrdbourke/FoodExtract-1k-Vision` (1,510 samples converted to LLaVA-style conversational format)
- **Fine-Tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT) via LoRA (frozen vision & LLM backbones)
- **Task:** Multimodal Structured Information Extraction (Image $\rightarrow$ Valid JSON)
- **Evaluation Strategy:** Schema validation, fuzzy set-based $F_1$, character similarity, and ROUGE-N/L overlap metrics.

### Target Schema
```json
{
  "is_food": 1,
  "image_title": "caprese salad",
  "food_items": ["tomatoes", "mozzarella", "basil"],
  "drink_items": []
}
```

---

## ⚙️ LoRA & Training Configuration

The model was fine-tuned for 1 epoch (48 global steps) using `bfloat16` precision and a cosine learning rate schedule with warmup.

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **PEFT Method** | LoRA |
| **LoRA Rank ($r$)** | 32 |
| **LoRA Alpha ($\alpha$)** | 64 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Precision** | `bfloat16` |
| **Optimizer / LR** | AdamW / Max $1 \times 10^{-4}$ (Cosine Decay) |

---

## 📊 Training Convergence & Curves

- **Initial Loss:** `1.7121`
- **Final Logged Loss:** `0.8888`
- **Minimum Loss:** `0.5107`
- **Total Loss Reduction:** **48.08%**

| Raw Training Loss Curve | Learning Rate Schedule |
| :---: | :---: |
| ![Loss Curve](eval/finetune_figures/training_loss_curve.png) | ![LR Schedule](eval/finetune_figures/learning_rate_schedule.png) |

---

## 📈 Benchmark & Evaluation Results

Evaluating structured text generation requires multi-level assessment. Exact string matching is overly strict due to natural language variability (e.g., ground truth *"caprese salad"* vs. prediction *"tomato salad"*). Therefore, fuzzy matching and token overlap metrics were used:

### 1. Schema & Sequence Evaluation
| Metric | 10 Samples | 50 Samples | Description |
| :--- | :---: | :---: | :--- |
| **JSON Parse Success Rate** | 100.0% | 98.0% | Measures valid JSON format generation |
| **`is_food` Accuracy** | 100.0% | 98.0% | Binary classification for food presence |
| **Title Similarity** | 0.606 | 0.578 | String-level character similarity (SequenceMatcher) |
| **ROUGE-1 / ROUGE-L** | *Evaluated* | *Evaluated* | Unigram & longest common subsequence overlap |

### 2. Attribute Extraction ($F_1$ Performance)
| Metric | 10 Samples | 50 Samples |
| :--- | :---: | :---: |
| **Food Items Precision** | 0.730 | 0.501 |
| **Food Items Recall** | 0.492 | 0.376 |
| **Food Items $F_1$ Score** | **0.582** | **0.421** |
| **Drink Items $F_1$ Score** | **0.900** | **0.893** |
| **Overall Item $F_1$** | **0.565** | **0.430** |

---

## 🚀 Quick Start & Inference

### 1. Environment Setup

```bash
pip install -U transformers accelerate peft qwen-vl-utils pillow pandas rouge-score
```

### 2. Run Inference

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "outputs/qwen25vl_foodextract_lora_full"

# Load base model & LoRA adapter
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
model.eval()

# Input image and prompt
image = Image.open("sample_food.jpg").convert("RGB")
prompt = """Analyze this image and return only valid JSON with exactly these keys:
- is_food
- image_title
- food_items
- drink_items"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3
    )

output = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:], 
    skip_special_tokens=True
)[0]

print(output)
```

---

## 📂 Repository Structure

```text
├── data/                                         # Processed LLaVA-style JSON datasets
│   ├── train_llava.json
│   └── val_llava.json
├── eval/                                         # Benchmark CSV logs & generated plots
│   └── finetune_figures/
│       ├── training_loss_curve.png
│       ├── training_loss_curve_smoothed.png
│       └── learning_rate_schedule.png
├── notebooks/                                    # Interactive pipelines
│   ├── foodextract_qwen_finetune.ipynb          # Data formatting & LoRA fine-tuning
│   ├── foodextract_qwen_evaluation.ipynb        # Inference, fuzzy F1, and ROUGE evaluation
│   └── qwen_finetuning_visualizations.ipynb     # Loss curves & configuration summary plots
├── outputs/                                      # Saved LoRA adapter weights & config
│   └── qwen25vl_foodextract_lora_full/
└── README.md
```
