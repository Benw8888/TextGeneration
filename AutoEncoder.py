import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import Encoder
import Decoder
import pickle as pickle
import gzip
import os

# Implement autoencoder, combining both encoder and decoder

class AutoEncoder(nn.Module):
    """
    Try to make an LSTM encoder
    """
    def __init__(self, encoder_file_path = None, decoder_file_path=None):
        super().__init__()
        self.encoder_file_path = encoder_file_path
        self.decoder_file_path = decoder_file_path
        if encoder_file_path is not None:
            # with gzip.open(encoder_file_path, 'rb') as inp:
            #     self.encoder = pickle.load(inp)
            self.load_encoder(encoder_file_path)
        else:
            self.encoder = Encoder.Encoder()
        self.decoder = Decoder.Decoder(file_path=decoder_file_path)

    def forward(self,input_ids=None, **kwargs):
        x = input_ids
        # x is in the format of a list of paragraphs
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

    def save(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        self.save_pretrained(target_folder)

    def save_pretrained(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        print("SAVING BASED ON: ", target_folder)
        torch.save(self.encoder.state_dict(), target_folder+"encoder.p")
        # with gzip.open(target_folder+"encoder.p", 'wb') as outp:
        #     pickle.dump(self.encoder, outp, -1)
        print("saving decoder!")
        self.decoder.save_pretrained(target_folder+"decoder")

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))



