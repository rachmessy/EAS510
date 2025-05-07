import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MoviePosterDataset(Dataset):
    def __init__(self, dataframe, poster_directory, transform=None):
        self.df = dataframe
        self.poster_directory = poster_directory
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = str(row['id'])
        label = torch.tensor(row['multi_hot_labels'], dtype=torch.float32)

        img_path = os.path.join(self.poster_directory, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    


class MultiModalDataset(Dataset):
    def __init__(self, df, poster_dir, transform=None):
        self.df = df
        self.poster_dir = poster_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.poster_dir, f"{row['id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_input = torch.tensor(row["input_ids"], dtype=torch.long)
        label = torch.tensor(row["multi_hot_labels"], dtype=torch.float32)
        return text_input, image, label


class MultiModalDataset2(Dataset):
    def __init__(self, df, poster_dir, transform=None):
        self.df = df
        self.poster_dir = poster_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.poster_dir, f"{row['id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        label = torch.tensor(row["multi_hot_labels"], dtype=torch.float32)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            image,
            label
        )

