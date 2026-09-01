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

## 🏗️ Architecture & Fine-Tuning Pipeline

The Vision-Language Model is adapted using parameter-efficient fine-tuning (PEFT/LoRA). Both the Vision Encoder and the LLM Backbone are frozen to conserve compute and prevent catastrophic forgetting, while low-rank adapters are injected across attention and MLP projection layers.

<p align="center">
  <img width="1367" height="665" alt="Ekran görüntüsü 2026-09-01 220928" src="https://github.com/user-attachments/assets/dc65a5c6-78a8-4bd2-9ca6-c53215a1f3c5" />

</p>

*Only LoRA / adapter parameters are updated during fine-tuning.*
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
## 🖼️ Qualitative Extraction Examples

Evaluation results show that exact string matching is overly punitive for Vision-Language Models[cite: 1]. While low $F_1$ scores might suggest model failure, inspecting ground truth versus predicted outputs reveals strong semantic alignment and valid visual parsing[cite: 1].

---

### Example 1: High Semantic & Beverage Matching (`218092.jpg` — $F_1$: 0.80)[cite: 1]

<p align="center">
  <img width="308" height="448" alt="image" src="https://github.com/user-attachments/assets/dcd65868-c7e4-4593-a0be-979ebdac36a4" />

 />
</p>

#### Ground Truth[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "French toast",
  "food_items": [
    "French toast",
    "strawberries",
    "powdered sugar",
    "butter"
  ],
  "drink_items": [
    "orange juice"
  ]
}
```

#### Model Prediction[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "fruit toast",
  "food_items": [
    "toast",
    "strawberries",
    "butter"
  ],
  "drink_items": [
    "orange juice"
  ]
}
```

> **Analysis:** The model captures the primary dish, key toppings (strawberries, butter), and correctly isolates the beverage (orange juice)[cite: 1].

---

### Example 2: Ingredient Identification (`3716522.jpg` — $F_1$: 0.75)[cite: 1]

<p align="center">
  <img width="484" height="391" alt="image" src="https://github.com/user-attachments/assets/a77e6ae4-98be-4df9-90d6-54173506494d" />
</p>

#### Ground Truth[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "caprese salad",
  "food_items": [
    "tomatoes",
    "mozzarella cheese",
    "basil",
    "balsamic glaze"
  ],
  "drink_items": []
}
```

#### Model Prediction[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "tomato salad",
  "food_items": [
    "tomatoes",
    "mozzarella",
    "basil"
  ],
  "drink_items": []
}
```

> **Analysis:** Rather than using the culinary name (*caprese salad*), the model describes the visual contents (*tomato salad*) while identifying core components (*tomatoes*, *mozzarella*, *basil*)[cite: 1].

---

### Example 3: Sub-component Matching (`2544128.jpg` — $F_1$: 0.71)[cite: 1]

<p align="center">
  <img width="404" height="448" alt="image" src="https://github.com/user-attachments/assets/11ec4d76-5f0f-47d7-be6d-ad4b63092a24" />

</p>

#### Ground Truth[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "lobster roll french fries",
  "food_items": [
    "lobster roll",
    "french fries",
    "mayonnaise",
    "lemon"
  ],
  "drink_items": []
}
```

#### Model Prediction[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "lobster roll",
  "food_items": [
    "lobster roll",
    "french fries"
  ],
  "drink_items": []
}
```

> **Analysis:** The model identifies the primary entree and side dish (*lobster roll*, *french fries*), omitting smaller garnish items (*lemon*, *mayonnaise*)[cite: 1].

---

### Example 4: Semantic Alignment vs. Low Token Overlap (`2001998.jpg` — $F_1$: 0.18)[cite: 1]


#### Ground Truth[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "burrito",
  "food_items": [
    "burrito",
    "tortilla",
    "beans",
    "salsa",
    "cheese",
    "rice"
  ],
  "drink_items": []
}
```

#### Model Prediction[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "pastry",
  "food_items": [
    "pastry",
    "filling"
  ],
  "drink_items": []
}
```

> **Analysis:** A low score ($F_1 = 0.18$) reflects ground truth labeling of interior ingredients not directly visible (*beans, rice, salsa*)[cite: 1]. The model relies strictly on exterior visual appearance (*pastry/wrapped dough with filling*), highlighting the limitation of set-overlap metrics against partially occluded contents[cite: 1].

---

### Example 5: Generic vs. Fine-Grained Description (`2366722.jpg` — $F_1$: 0.20)[cite: 1]



#### Ground Truth[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "seared scallops dish",
  "food_items": [
    "seared scallops",
    "sauce",
    "garnish",
    "pea puree"
  ],
  "drink_items": []
}
```

#### Model Prediction[cite: 1]
```json
{
  "is_food": 1,
  "image_title": "scallops",
  "food_items": [
    "scallops"
  ],
  "drink_items": []
}
```

> **Analysis:** The model detects the main protein (*scallops*) correctly, but omits plated accents (*pea puree, garnish*), causing a low metric score despite valid core identification[cite: 1].
## 🖼️ Qualitative Extraction Examples

Comparison between **Ground Truth** annotations and **Fine-tuned Qwen2.5-VL** structured outputs:

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
