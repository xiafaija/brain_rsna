from torch.utils import data as torch_data
import pydicom
import cv2
import os
import glob
import numpy as np
import torch

import random
import pandas as pd
from sklearn import model_selection as sk_model_selection
import matplotlib.pyplot as plt

train_df = pd.read_csv("F:/rsna_data/train_labels.csv")
df = pd.read_csv("F:/rsna_data/train_labels.csv")

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

df_train, df_valid = sk_model_selection.train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["MGMT_value"],
)


class DataRetriever(torch_data.Dataset):
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = f"F:/rsna_data/train/{str(_id).zfill(5)}/"
        channels = []
        for t in ("FLAIR", "T1w", "T1wCE"):  # "T2w"
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            # start, end = int(len(t_paths) * 0.475), int(len(t_paths) * 0.525)
            x = len(t_paths)
            if x < 10:
                r = range(x)
            else:
                d = x // 10
                r = range(d, x - d, d)

            channel = []
            # for i in range(start, end + 1):
            for i in r:
                channel.append(cv2.resize(load_dicom(t_paths[i]), (256, 256)) / 255)
            channel = np.mean(channel, axis=0)
            channels.append(channel)

        y = torch.tensor(self.targets[index], dtype=torch.float)

        return {"X": torch.tensor(channels).float(), "y": y}

train_data_retriever = DataRetriever(
    df_train["BraTS21ID"].values,
    df_train["MGMT_value"].values,
)

valid_data_retriever = DataRetriever(
    df_valid["BraTS21ID"].values,
    df_valid["MGMT_value"].values,
)

def train_loader():

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    return train_loader

def valid_loader():
    valid_loader = torch_data.DataLoader(
        valid_data_retriever,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    return valid_loader