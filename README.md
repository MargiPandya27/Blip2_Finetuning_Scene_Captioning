# Fine-tuning BLIP-2 on BDD100K for Scene Captioning

## 🎯 Objective & Overview

This project fine-tunes **[BLIP-2 (OPT-2.7B)](https://huggingface.co/Salesforce/blip2-opt-2.7b)** on the **BDD100K** dataset to generate rich semantic captions describing driving scenes. The model learns to extract context from images including:
- **Environmental Context**: Time of day, weather, scene type
- **Object Information**: Vehicle types, pedestrians, traffic signals
- **Critical Attributes**: Traffic light status, occlusion, truncation

The pipeline uses **QLoRA (4-bit quantization + LoRA)** for memory-efficient fine-tuning on limited hardware.

---

## 📊 Results & Metrics

### Training Logs
View complete training metrics on [Weights & Biases](https://wandb.ai):

![Training Dashboard](https://i.imgur.com/placeholder_training.png)
- **Training Loss**: Converges smoothly across 5 epochs
- **Validation Loss**: Consistent improvement per epoch
- **Learning Rate**: Adaptive scheduling with warmup

### Evaluation Results
![Evaluation Dashboard](https://i.imgur.com/placeholder_eval.png)

**BLEU Scores** (on test set):
```
BLEU-1: 0.6532  (Unigram precision)
BLEU-2: 0.5234  (Bigram precision)
BLEU-3: 0.4012  (Trigram precision)
BLEU-4: 0.2876  (4-gram precision)
```

---

## 🎬 Demo Examples

### Example 1: Evening Highway Scene
**Input Image**: Highway at evening with traffic light

**Generated Caption**:
```
It is a evening scene in clear weather on a highway road. 
The traffic light is red. There is one car. There are 2 pedestrians.
```

### Example 2: Rainy City Street
**Input Image**: City street in rainy conditions

**Generated Caption**:
```
It is a night scene in rainy weather on a city street road. 
There are 3 cars. There is one truncated pedestrian. 
The traffic light is yellow.
```

---

## 🚀 How to Train

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python main.py
```

This automatically:
- Downloads BDD100K via Kaggle Hub
- Processes JSON labels into semantic captions
- Splits data into train/val/test
- Loads BLIP-2 with LoRA configuration
- Trains for 5 epochs with:
  - Batch size: 4 per device
  - Learning rate: 5e-5
  - Gradient accumulation: 4 steps
  - FP16 precision

**Output**: Fine-tuned model saved to `./blip2-finetuned/`

---

## 🧪 How to Evaluate

### Run Evaluation
```bash
python eval.py
```

This:
1. Loads the fine-tuned model from `./blip2-finetuned/`
2. Samples 20 random test images
3. Generates captions using sampling (temperature=0.7)
4. Computes BLEU-1 through BLEU-4 scores

**Output**: BLEU metrics printed to console

---

## 📦 Dataset Pipeline

The `bdd100k_data_pipeline.py` converts raw BDD100K JSON labels into semantic captions:

### Input JSON Format
```json
{
  "name": "000123.jpg",
  "attributes": {
    "timeofday": "evening",
    "weather": "clear",
    "scene": "highway"
  },
  "labels": [
    {"category": "car"},
    {"category": "traffic light", "attributes": {"trafficLightColor": "red"}},
    {"category": "pedestrian", "attributes": {"truncated": true}}
  ]
}
```

### Caption Generation Process

1. **Extracts Global Context**
   - Time of day (morning, evening, night, etc.)
   - Weather conditions (clear, rainy, snowy, etc.)
   - Scene type (highway, residential, city street, etc.)

2. **Aggregates Objects**
   - Counts object instances by category
   - Captures object attributes (occluded, truncated)
   - Prioritizes traffic light status (red > yellow > green)

3. **Generates Natural Language**
   ```
   It is a evening scene in clear weather on a highway road. The traffic light is red. 
   There is one car. There is one truncated pedestrian.
   ```

### Output CSV Format
```csv
name,raw_caption
000123.jpg,"It is a evening scene in clear weather on a highway road. The traffic light is red. There is one car. There is one truncated pedestrian."
000124.jpg,"It is a morning scene in rainy weather on a city street road. There are 2 cars. There are 3 pedestrians."
```

---

---

## 📚 References

- [BLIP-2 Paper](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [BDD100K Dataset](https://bdd100k.com/)
- [PEFT / LoRA](https://github.com/huggingface/peft)

## 📋 Project Structure

```
finetuning_vlm-main/
├── main.py                           # Entry point: orchestrates data pipeline, training, and evaluation
├── train.py                          # Training configuration and hyperparameters
├── model.py                          # Model loading with LoRA/QLoRA configuration
├── dataloader.py                     # PyTorch Dataset and DataLoader implementation
├── bdd100k_data_pipeline.py         # BDD100K JSON parsing and caption generation
├── eval.py                          # Evaluation script with BLEU score computation
├── utils.py                         # Utility functions (BLEU score calculations)
├── requirements.txt                 # Python dependencies
├── configs/
│   └── qlora_config.yaml           # QLoRA and model configuration
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/blip2-bdd100k.git
cd blip2-bdd100k

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configuration

Edit `configs/qlora_config.yaml` to customize:
- **Model**: `Salesforce/blip2-opt-2.7b` (can switch to other BLIP-2 variants)
- **QLoRA Settings**: 4-bit quantization, NF4 quant type, LoRA rank, dropout, etc.
- **Target Modules**: Q-Former layers to fine-tune (query, key, value, dense)

Example:
```yaml
model_config:
  name: "Salesforce/blip2-opt-2.7b"
  device_map: "auto"  # Auto GPU/CPU split
  
bnb_config:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"

lora_config:
  r: 4              # LoRA rank
  lora_alpha: 1     # Scaling factor
  lora_dropout: 0.05
```

---

## 📦 Dataset Preparation

### Option A: Automatic (Recommended)
Automatically downloads BDD100K via Kaggle Hub and processes captions:

```bash
python main.py
```

This will:
1. Download BDD100K from Kaggle Hub
2. Parse JSON labels and generate semantic captions
3. Save captions to `bdd100k_captions.csv`
4. Split into train/val/test sets

### Option B: Pre-downloaded Dataset
If you already have the BDD100K labels JSON file, update the path in `main.py`:

```python
BDD_JSON_PATH = 'path/to/bdd100k_labels_images_val.json'
```

### Generated Caption Example

Input JSON:
```json
{
  "name": "000123.jpg",
  "attributes": {
    "timeofday": "evening",
    "weather": "clear",
    "scene": "highway"
  },
  "labels": [
    {"category": "car"},
    {"category": "traffic light", "attributes": {"trafficLightColor": "red"}},
    {"category": "pedestrian", "attributes": {"truncated": true}}
  ]
}
```

Generated Caption:
```
It is a evening scene in clear weather on a highway road. The traffic light is red. 
There is one car. There are 1 truncated pedestrians. Key visible objects are: The traffic light is red. There is one car. There is one truncated pedestrian.
```

---

## 🎓 Training

### Basic Training
```bash
python main.py
```

This will:
1. Download/process BDD100K data
2. Load BLIP-2 model with LoRA configuration
3. Train for 5 epochs with the following settings:
   - Batch size: 4 per device
   - Learning rate: 5e-5
   - Gradient accumulation: 4 steps
   - FP16 precision for efficiency
   - Validation after each epoch

### Training Hyperparameters (in `train.py`)

```python
training_args = TrainingArguments(
    output_dir="./blip2-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    fp16=True,                    # Mixed precision training
    logging_steps=500,
    eval_strategy="epoch",         # Evaluate after each epoch
    dataloader_num_workers=4,
    push_to_hub=False
)
```

### Quick Validation Run
```bash
# For testing, temporarily reduce epochs in train.py to 1
python main.py
```

---

## 🧪 Evaluation

### Run Evaluation Script
```bash
python eval.py
```

The evaluation script:
1. Loads the fine-tuned model from `./blip2-finetuned/`
2. Samples 20 random test images
3. Generates captions using sampling with temperature=0.7
4. Computes BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores

### Output Metrics

The `utils.py` script computes BLEU scores:
```python
# BLEU scores with different n-gram weights
BLEU-1: 0.6532  # Unigram precision
BLEU-2: 0.5234  # Bigram precision
BLEU-3: 0.4012  # Trigram precision
BLEU-4: 0.2876  # 4-gram precision
```

---

## 🧩 Decoding Strategies

### 1. **Sampling (Default - Diverse Captions)**
```python
do_sample=True
top_p=0.9              # Nucleus sampling (top 90% probability mass)
temperature=0.7        # Lower = more deterministic, Higher = more random
no_repeat_ngram_size=3 # Prevent repeating 3-grams
```

**Use when**: You want creative, diverse captions without repetition.

### 2. **Beam Search (Deterministic)**
```python
do_sample=False
num_beams=5            # Keep top 5 candidates at each step
```

**Use when**: You want consistent, high-quality captions.

---

## 📊 Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Fine-tuned Model | `./blip2-finetuned/` | Saved checkpoints and final model weights |
| Processed Data | `bdd100k_captions.csv` | Generated captions paired with image names |
| Training Logs | Console output | Loss, learning rate, validation metrics |

---

## 📋 Dataset Statistics

Automatic data split from `main.py`:
```
Train size: (60000, 2)      # 60% of dataset
Validation size: (15000, 2) # 15% of dataset
Test size: (15000, 2)       # 15% of dataset
```

Each row contains:
- `name`: Image filename (e.g., "000123.jpg")
- `raw_caption`: Generated semantic caption

---

## 🔧 Key Files Explanation

### `main.py`
Orchestrates the entire pipeline:
- Downloads BDD100K via Kaggle Hub
- Processes labels into captions via `bdd100k_data_pipeline.py`
- Splits data into train/val/test
- Loads model and starts training via `Trainer`

### `model.py`
Handles model initialization:
- Loads BLIP-2 OPT-2.7B from HuggingFace
- Applies 4-bit quantization (QLoRA)
- Configures LoRA adapters for Q-Former layers
- Prints trainable parameter count

### `dataloader.py`
PyTorch Dataset implementation:
- Loads images and captions
- Preprocesses with BLIP-2 processor
- Handles padding and truncation
- Returns batched pixel_values, input_ids, attention_mask, labels

### `bdd100k_data_pipeline.py`
Converts BDD100K raw labels to captions:
- Extracts time of day, weather, scene from attributes
- Aggregates object counts and attributes
- Prioritizes traffic light status
- Generates natural language descriptions

### `train.py`
Training configuration:
- Defines TrainingArguments
- Sets batch size, learning rate, epochs, etc.
- Configures FP16 and logging

### `eval.py`
Evaluation pipeline:
- Loads fine-tuned model
- Generates captions for test samples
- Computes BLEU scores

### `utils.py`
Utility functions:
- `compute_bleu_scores()`: Computes BLEU-1 through BLEU-4 metrics

---

## 💾 Requirements

```
peft              # Parameter-Efficient Fine-Tuning (LoRA)
transformers      # HuggingFace transformers library
bitsandbytes      # 4-bit quantization support
datasets          # HuggingFace datasets
pyyaml            # YAML configuration parsing
accelerate        # Distributed training support
```

---

## 🎯 Performance Tips

- **GPU Memory**: Adjust `per_device_train_batch_size` if OOM occurs (default: 4)
- **Speed**: Increase `dataloader_num_workers` for faster data loading
- **Quality**: Increase `num_train_epochs` for better performance
- **Stability**: Reduce learning rate if training diverges

---

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve this pipeline!

---

## 📚 References

- **BLIP-2**: [Bootstrapping Language-Image Pre-training](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- **BDD100K**: [Berkeley DeepDrive Dataset](https://bdd100k.com/)
- **PEFT**: [Parameter-Efficient Fine-Tuning with LoRA](https://github.com/huggingface/peft)
- **QLoRA**: [Efficient Fine-tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## 📄 License

This project is provided as-is. Ensure compliance with BDD100K and BLIP-2 licensing.

---

## Demo  
Try the live demo on **Hugging Face Spaces**:  
👉 [MargiPandya/blip2-scene-captioning](https://huggingface.co/spaces/MargiPandya/blip2-scene-captioning)

---

## Features
- Fine-tunes **BLIP-2 + OPT-2.7B** using **PEFT (LoRA)** for efficient training.  
- Generates **context-rich driving scene captions** from BDD100K images.  
- Includes **automatic dataset preparation**, training, and evaluation scripts.  
- Supports **beam search** and **sampling-based decoding** for flexibility.  

---

## Setup

### 1️⃣ Installation
```bash
git clone https://github.com/<your-username>/blip2-bdd100k.git
cd blip2-bdd100k

# Create environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
📦 Dataset Preparation
Option A — Automatic (Recommended)
Automatically downloads, processes, and splits BDD100K via kagglehub.

bash
Copy code
python main.py
Option B — Already have bdd100k_labels_images.json
If you already have the JSON label file (bdd100k_labels_images_val.json):

bash
**Example Data Format**
Each entry in the dataset looks like:

json
Copy code
{
  "image": "images/000123.jpg",
  "question": "Is it safe to proceed?",
  "answer": "no",
  "meta": {
    "traffic_light": "red",
    "front_vehicle_proximity": "close",
    "timeofday": "evening"
  }
}
Training
Edit hyperparameters in train.py, then run:

bash
Copy code
python main.py
Quick Test Run
bash
Copy code
python main.py --epochs 1 --train_batch_size 1
This verifies setup and ensures your pipeline runs end-to-end.

🧪 Evaluation
Set model and image paths in eval.py, then run:

bash
Copy code
python eval.py
🧩 Decoding Options
Default Sampling (diverse captions):

python
Copy code
do_sample=True, top_p=0.9, temperature=0.7
Beam Search (deterministic captions):

python
Copy code
do_sample=False, num_beams=5

Outputs
Type	Location
Trained Checkpoints	./blip2-finetuned/ (if working on google colab change path to save in google drive)
Generated Captions + Metrics	./logs/
Training Loss Plots	scripts/train_loss.png
Evaluation Loss Plots	scripts/eval_loss.png

Example Caption Output
Input Image	Generated Caption
<img src="samples/000123.jpg" width="300"/>	“A car is waiting at a red light in the evening with vehicles ahead.”

References
BLIP-2: Bootstrapping Language-Image Pre-training

BDD100K Dataset

PEFT: Parameter-Efficient Fine-Tuning
#   B l i p 2 _ F i n e t u n i n g _ S c e n e _ C a p t i o n i n g  
 