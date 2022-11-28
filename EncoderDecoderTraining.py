import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import Encoder
import Decoder
import AutoEncoder
import os
import shutil
import subprocess
import sys

from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase

# Implement autoencoder, combining both encoder and decoder

class WrappedAutoEncoder(pl.LightningModule):
    """
    Autoencoder wrapped for pl lightning training
    """
    def __init__(self, model, dataset, hparams, tokenizer):
        super().__init__()
        # define word embedding module
        self.model, self.dataset, self.tokenizer = (
            model,
            dataset,
            tokenizer,
        )
        self.save_hyperparameters(hparams)

    def forward(self,x):
        return self.model.forward(x)

    def training_step(self, batch_inp, step_id):
        x = batch_inp
        outputs = self.forward(x)
        loss = outputs[0]
        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=self.hparams["pin_memory"],
            num_workers=self.hparams["num_workers"],
        )

    def configure_optimizers(self):
        "Prepare optimizer"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.hparams["num_steps"],
        )

        return [optimizer], [scheduler]