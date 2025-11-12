from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image"]).convert("RGB")

        # ✅ Ensure caption is a valid string
        caption = item["text"]
        if not isinstance(caption, str):
            caption = "" if pd.isna(caption) else str(caption)

        # Process both image and text
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=114
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = input_ids.clone()  # For captioning loss

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
