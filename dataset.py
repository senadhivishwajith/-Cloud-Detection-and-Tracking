import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CloudMaskSequenceDataset(Dataset):
    def __init__(self, csv_path, image_folder, sequence_length=30, image_size=(64, 64)):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.image_folder = image_folder

        # Load CSV and filter for clouds
        df = pd.read_csv(csv_path)
        df = df[df["type"] == "cloud"]
        df = df.sort_values(by=["object_id", "frame"])

        # Group into valid sequences
        self.sequences = []
        for cloud_id, group in df.groupby("object_id"):
            if len(group) >= sequence_length:
                frames = group["filename"].tolist()
                for i in range(len(frames) - sequence_length + 1):
                    self.sequences.append(frames[i:i + sequence_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_files = self.sequences[idx]
        masks = []

        for fname in sequence_files:
            img_path = os.path.join(self.image_folder, f"tracked_{fname}")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing: {img_path}")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_NEAREST)
            img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            masks.append(img_tensor)

        # Split input/target
        input_seq = torch.stack(masks[:-1])   # shape: (seq_len-1, 1, H, W)
        target = masks[-1]                    # shape: (1, H, W)

        return {
            "input_sequence": input_seq,
            "target_mask": target
        }

