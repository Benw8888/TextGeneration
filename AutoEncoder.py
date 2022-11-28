import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import Encoder
import Decoder

# Implement autoencoder, combining both encoder and decoder

class AutoEncoder(nn.Module):
    """
    Try to make an LSTM encoder
    """
    def __init__(self):
        super().__init__()
        # define word embedding module:
        self.encoder = Encoder.Encoder()
        self.decoder = Decoder.Decoder()
        self.decoder_word_embeddings = self.decoder.word_embeddings

    def forward(self,x):
        paragraph_embedding = self.encoder(x)
        # Now we feed in x and paragraph_embedding and hope to get x back
        recovered_x = self.decoder(paragraph_embedding, x)

        return recovered_x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        # Shouldn't be used, use WrappedAutoEncoder instead
        x = batch_inp
        outputs = self.forward(x)
        loss = outputs[0]
        return {"loss": loss}

