import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from config import get_config

# Load configuration
args = get_config()

# Load GO terms and GO embeddings
go_SS_embed = torch.load(args.go_SS_embdding)
go_id = go_SS_embed["go_id"]

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

# Load and process annotations
def process_annotations(data):
    annotations = {}
    for idx, row in data.iterrows():
        annotations[row["proteins"]] = row["annotations"]
    return annotations

train_data = load_data(args.train_data_dir)
test_data = load_data(args.test_data_dir)
train_annotations = process_annotations(train_data)
test_annotations = process_annotations(test_data)

# Process DIAMOND predictions
def process_diamond(file_path):
    diamond_preds = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            protein, score = parts[0], float(parts[2])
            diamond_preds.setdefault(protein, []).append(score)
    return diamond_preds

diamond_preds = process_diamond(args.diamond_dir)

# Compute combined predictions
def compute_combined_predictions(annotations, diamond_preds, go_id, existing_preds, alpha):
    combined_preds = []
    for protein, _ in annotations.items():
        if protein in diamond_preds:
            weights = np.array(diamond_preds[protein]) / np.sum(diamond_preds[protein])
            combined_pred = np.sum(weights[:, None] * existing_preds, axis=0)
        else:
            combined_pred = existing_preds
        combined_preds.append(combined_pred * alpha + existing_preds * (1 - alpha))
    return np.array(combined_preds)

alpha = 0.4
existing_preds = np.load('Data/CAFA3/test/MFO/MFO_pred_epoch19.npy')
combined_preds = compute_combined_predictions(test_annotations, diamond_preds, go_id, existing_preds, alpha)

# Save the combined predictions
np.save('Data/CAFA3/test/MFO/MFO_pred_combine_diamond_epoch19.npy', combined_preds)
