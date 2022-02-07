import numpy as np
import pytorch_lightning as pl
from pytorch_metric_learning import losses
import torch

from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError

from constants import DATA_SPLITS
from utils import evaluate_solution


class AEModel(pl.LightningModule):
    def __init__(self, args, solution_ad):
        super().__init__()
        layer_sizes = [2**(9 - i) * args.layer_size_multiplier for i in range(args.n_hidden)]
        layer_sizes = np.array([0] + layer_sizes + [0]).astype(int)
        layer_sizes[-1] = args.embedding_dim
        layer_sizes[0] = args.expression_dim
        self.info_nce_weight = args.info_nce_weight

        self.mse = MeanSquaredError(dist_sync_on_step=True)
        self.contrastive_loss = losses.ContrastiveLoss(pos_margin=1, neg_margin=0)

        STARTING_LOWEST_MSE = 9999
        self.lowest_mse = STARTING_LOWEST_MSE

        modules = []
        # encoder
        for i in range(len(layer_sizes) - 2):
            cur_layer = nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.SELU(),
                nn.Dropout(args.dropout_p))
            if args.batchnorm:
                cur_layer = nn.Sequential(cur_layer, nn.BatchNorm1d(layer_sizes[i + 1]))
            modules.append(cur_layer)
        final_layer = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        modules.append(final_layer)
        self.encoder = nn.ModuleList(modules)

        self.solution_ad = solution_ad
        self.run_name = args.run_name

        modules = []
        # decoder
        for i in range(len(layer_sizes) - 2):
            ind = len(layer_sizes) -2 - i
            cur_layer = nn.Sequential(
                nn.Linear(layer_sizes[ind + 1], layer_sizes[ind]), nn.SELU(),
                nn.Dropout(args.dropout_p))
            if args.batchnorm:
                cur_layer = nn.Sequential(cur_layer, nn.BatchNorm1d(layer_sizes[ind + 1]))
            modules.append(cur_layer)
        final_layer = nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[0]))
        modules.append(final_layer)
        self.decoder = nn.ModuleList(modules)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch["expression"], return_encoding=True)

    def _epoch_end_logging(self, preds, target, split, log_append=""):
        cur_mse = self.mse(preds, target).item()
        if log_append != "":
            log_append = "_" + log_append
        self.log(f"mse_{split}{log_append}", cur_mse)
        return cur_mse

    def forward(self, x, return_encoding=False):
        out = x
        for layer in self.encoder:
            out = layer(out)
        emb = out.clone()
        for layer in self.decoder:
            out = layer(out)
        if return_encoding:
            return out, emb
        return out

    def _epoch_end(self, outputs, split):
        epoch_preds = torch.cat([x["preds"] for x in outputs])
        epoch_target = torch.cat([x["target"] for x in outputs])
        epoch_loss = torch.Tensor([x["loss"] for x in outputs])
        self.log(f"loss_{split}", epoch_loss.mean().item())

        cur_mse = self._epoch_end_logging(epoch_preds, epoch_target, split)
        if split == DATA_SPLITS[1] and cur_mse < self.lowest_mse:
            self.lowest_mse = cur_mse
            self.log(f"best_mse_{split}", cur_mse)

    def _step(self, batch):
        preds, emb = self(batch["expression"], return_encoding=True)
        loss_ae = F.mse_loss(preds, batch["expression"])
        loss_nce = self.contrastive_loss(emb, batch["batch_label"])
        out = {"loss": loss_ae - loss_nce * self.info_nce_weight, "preds": preds.detach(), "target": batch["expression"]}
        return out

    def training_step(self, batch, _):
        return self._step(batch)

    def test_step(self, batch, _):
        return self._step(batch)

    def validation_step(self, batch, _):
        emb = self(batch["expression"], return_encoding=True)[1]
        out = {"emb": emb}
        return out

    def validation_epoch_end(self, outputs):
        epoch_embs = torch.cat([x["emb"] for x in outputs]).cpu().numpy()
        for key, value in evaluate_solution(self.solution_ad, epoch_embs, self.run_name).items():
            self.log(key, value)
        return


    def training_epoch_end(self, training_step_outputs):
        self._epoch_end(training_step_outputs, DATA_SPLITS[0])

    def test_epoch_end(self, test_step_outputs):
        return self._epoch_end(test_step_outputs, DATA_SPLITS[2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
        return optimizer