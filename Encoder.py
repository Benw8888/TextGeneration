import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Implement lstm encoder


class Encoder(pl.LightningModule):
    """
    Try to make an LSTM encoder
    """
    def __init__(self, level="paragraph"):
        super().__init__()
        # define word embedding module:
        embedding_dim = 512
        hidden_dim = 768
        output_dim = 768
        self.embedding = nn.Embedding(num_embeddings=50257,embedding_dim=embedding_dim, dtype=torch.float64)
        # define lstm module:
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, bidirectional=True, dtype=torch.float64, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2,output_dim, dtype=torch.float64)

    def forward(self,x):
        embedded_x = self.embedding(x)
        lstm_outputs, _ = self.lstm(embedded_x.to(torch.float64))
        return self.linear(lstm_outputs[:,-1]) # pass final hidden states into the linear module

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass




