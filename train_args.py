from transformers import TrainingArguments, Trainer
import torch

training_args = TrainingArguments(
    output_dir="./blip2-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    fp16=True,
    logging_steps=500,
    eval_strategy="epoch",    # Evaluate every `eval_steps`
    # eval_steps=100,                # Adjust as needed
    # save_steps=100,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    label_names=["labels"],
    push_to_hub=False
)
