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

![Training Dashboard](https://github.com/MargiPandya27/Blip2_Finetuning_Scene_Captioning/blob/main/logs/eval_train_logs.png)
- **Training Loss**: Converges smoothly across 5 epochs
- **Validation Loss**: Consistent improvement per epoch
- **Learning Rate**: Adaptive scheduling with warmup

---

## Demo Examples

### Example 1: 
**Input Image**: 
![Rainy City Street](https://github.com/MargiPandya27/Blip2_Finetuning_Scene_Captioning/blob/main/logs/demo1.png)

**Generated Caption**:
```
It is a **daytime** scene in **clear** weather on a **city street** road. Key visible objects are: There are **3 cars**. There is one **drivable area**. There are **3 lanes**. There are **4 partially occluded cars**. There is one **partially occluded traffic sign**. There is one **partially occluded car**. There is one **partially occluded traffic sign**. 
```

### Example 2: 
**Input Image**: 
![Rainy City Street](https://github.com/MargiPandya27/Blip2_Finetuning_Scene_Captioning/blob/main/logs/demo2.png)

**Generated Caption**:
```
It is a **night** scene in **clear** weather on a **city street** road. Key visible objects are: There are **2 cars**. There is one **drivable area**. There are **2 lanes**. There are **2 partially occluded cars**. There is one **partially occluded traffic sign**. There is one **partially occluded traffic sign**. There is one **partially occluded traffic sign**. 
```

---

## Training

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

## Evaluation

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

## 📚 References

- [BLIP-2 Paper](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [BDD100K Dataset](https://bdd100k.com/)
- [PEFT / LoRA](https://github.com/huggingface/peft)
