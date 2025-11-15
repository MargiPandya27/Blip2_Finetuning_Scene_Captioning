import kagglehub
from bdd100k_data_pipeline import process_bdd_labels
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import ImageCaptioningDataset, collate_fn
from train_args import training_args
from transformers import TrainingArguments, Trainer
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch
import yaml
import os
from peft import LoraConfig, get_peft_model


def train():
    # Download latest version
    path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")

    print("Path to dataset files:", path)

    # --- Configuration ---
    # NOTE: Replace this path with the actual path to your BDD-100K labels file
    BDD_JSON_PATH = path + '/' + 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    # --- New Configuration for Output ---
    OUTPUT_CSV_PATH = 'bdd100k_captions.csv'

    # --- Execute the main processing function ---
    processed_dataset = process_bdd_labels(BDD_JSON_PATH)

    # --- CSV Saving Step ---
    if processed_dataset:
        # 1. Convert the list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(processed_dataset)

        # 2. Save the DataFrame to a CSV file
        # index=False prevents pandas from writing the DataFrame index as a column
        df.to_csv(OUTPUT_CSV_PATH, index=False)

        print(f"\n✅ Successfully processed {len(df)} records.")
        print(f"File saved to: **{OUTPUT_CSV_PATH}**")
    else:
        print("\n⚠️ No data was processed, so no CSV file was created.")


    df = pd.read_csv('bdd100k_captions.csv')

    # 70% train, 30% temp (for val + test)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)

    # Split temp into 70% val, 30% test
    val_df, test_df = train_test_split(temp_df, test_size=0.3, random_state=42, shuffle=True)

    # Show the sizes
    print(f"Train size: {train_df.shape}")
    print(f"Validation size: {val_df.shape}")
    print(f"Test size: {test_df.shape}")


    image_path = path + '/bdd100k/bdd100k/images/100k/val'

    # Create list of dictionaries
    train_dataset_list = [
        {"image": f"{image_path}/{row['name']}", "text": row['raw_caption']}
        for idx, row in train_df.iterrows()
    ]
    val_dataset_list = [
        {"image": f"{image_path}/{row['name']}", "text": row['raw_caption']}
        for idx, row in val_df.iterrows()
    ]

    test_dataset_list = [
        {"image": f"{image_path}/{row['name']}", "text": row['raw_caption']}
        for idx, row in test_df.iterrows()
    ]

    processor, model =  load_model()

    # This will now show only the parameters added to the Q-Former layers
    model.print_trainable_parameters()


    train_dataset = ImageCaptioningDataset(train_dataset_list, processor)
    val_dataset= ImageCaptioningDataset(val_dataset_list, processor)
    test_dataset = ImageCaptioningDataset(test_dataset_list, processor)

    # max_char_len = train_df['raw_caption'].apply(lambda x: len(x.split())).max()#.apply(len).max()
    # print(f"Maximum character length of training captions: {max_char_len}")
        
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    train()
