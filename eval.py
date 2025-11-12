import os
import random
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import compute_bleu_scores
from model import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForVision2Seq 

# Assuming test_df, processor, model already defined
# Also assume test_df has a column "caption" with the ground truth

def eval(image_folder='path_to_images'):

    df = pd.read_csv('bdd100k_captions.csv')

    # 70% train, 30% temp (for val + test)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)

    # Split temp into 70% val, 30% test
    val_df, test_df = train_test_split(temp_df, test_size=0.3, random_state=42, shuffle=True)

    # Show the sizes
    print(f"Train size: {train_df.shape}")
    print(f"Validation size: {val_df.shape}")
    print(f"Test size: {test_df.shape}")
    # Select 20 random samples
    sample_df = test_df.sample(n=20).reset_index(drop=True)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    references = []
    hypotheses = []
    # Replace AutoModelForSequenceClassification with your specific model class

    # Specify the directory where Trainer saved the model
    MODEL_PATH = "/content/blip2-finetuned/runs" 

    # Load the Tokenizer
    processor = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load the Model (the fine-tuned weights and config)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH) 
    model.to(device)

# Move the model to a GPU if available
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The loaded_model is now ready for inference!

    for idx in range(20):
        image_filename = sample_df.loc[idx, "name"]
        full_image_path = os.path.join(image_folder, image_filename)

        # Open and preprocess the image
        img = Image.open(full_image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=75,
            do_sample=True,          # **Enable sampling**
            top_p=0.9,               # **Top-p sampling (e.g., 90% probability)**
            # top_k=50,              # OR Top-k sampling (choose one)
            temperature=0.7,         # **Lower temp for stability/less randomness**
            num_beams=1,             # Use 1 for sampling, or > 1 for Beam Search
            no_repeat_ngram_size=3,  # **Penalty for repeating 3-grams**
        )
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Tokenize
        reference_caption = sample_df.loc[idx, "raw_caption"]  # adjust if multiple captions exist
        references.append([reference_caption.lower().split()])  # list of reference tokens
        hypotheses.append(generated_caption.lower().split())    # list of hypothesis tokens


if __name__ == "__main__":
    image_folder = "/kaggle/input/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/100k/val"
    eval(image_folder)