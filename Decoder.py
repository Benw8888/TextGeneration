import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Implement gpt2 decoder conditioned on paragraph embedding
# most of this code is bad


class Decoder(pl.LightningModule):
    """
    Try to make an LSTM encoder
    """
    def __init__(self, level="paragraph"):
        super().__init__()
        # define word embedding module:
        self. word_embeddings = None

        # create custom modified GPT2LMHeadModel


    def forward(self,embedding, x):
        return self.model(input_ids= x, labels=x, conditioned_on=embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass




