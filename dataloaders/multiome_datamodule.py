import os
import numpy as np

import pytorch_lightning as pl
import torch

from typing import Optional

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import load_dataset


class MultiomeDataset(Dataset):
    def __init__(self, expression, batch, obs_name, batch_label):
        self.expression = expression
        self.batch = batch
        self.obs_name = obs_name
        self.batch_label = batch_label

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, idx):
        return {"expression": np.asarray(self.expression[idx].todense()).squeeze(-2),
                "batch_label": self.batch_label[idx]}



class MultiomeDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size, minmax_norm, std_norm):
        super().__init__()
        self.dataset = load_dataset(path=data_path)
        ad_mod1, _, adata_solution = self.dataset
        expression = ad_mod1.X

        if std_norm == True:
            scaler = StandardScaler()
            scaler.fit(expression)
            expression = scaler.transform(expression)
        elif minmax_norm == True:
            scaler = MinMaxScaler()
            scaler.fit(expression)
            expression = scaler.transform(expression)
        batches = ad_mod1.obs.batch.str.split("d", expand=True)
        site, donor = batches[0].apply(lambda x: int(x[1])).values - 1, batches[1].apply(int).values - 1
        donor = np.where(donor == 9, 7, donor)
        sites, donors = np.unique(site), np.unique(donor)
        obs_names = ad_mod1.obs.index.to_numpy()

        batch_one_hot = np.zeros((len(site), len(sites) + len(donors)), dtype="float32")
        batch_one_hot[np.arange(len(batch_one_hot)), site] = 1
        batch_one_hot[np.arange(len(batch_one_hot)), len(sites) + donor] = 1

        self.dataset_params = {
            "expression": expression,
            "batch": batch_one_hot,
            "obs_name": obs_names,
            "batch_label": batches[0].astype("category").cat.codes.values
        }

        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.multiome_dataset = MultiomeDataset(**self.dataset_params)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.multiome_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(),
                          pin_memory=True if torch.cuda.is_available() else False, shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self.train_dataloader(shuffle=False)

    def predict_dataloader(self):
        return self.train_dataloader(shuffle=False)
