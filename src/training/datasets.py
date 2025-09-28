import json
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageCSVDataset(Dataset):
    def __init__(self, csv_path: str, classes_json: str, transform: Callable = None):
        self.df = pd.read_csv(csv_path)
        with open(classes_json, "r", encoding="utf-8") as f:
            self.class_to_idx: Dict[str, int] = json.load(f)
        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["file_path"]
        label_name = row["label"]
        y = self.class_to_idx[str(label_name)]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)
