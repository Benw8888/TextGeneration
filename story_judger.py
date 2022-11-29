import numpy as np
import torch
import torch.nn as nn

# Implement story judger


class BaselineStoryJudger(nn.Module):
    """
    Try to make a baseline story-judger using an LSTM.
    """
    def __init__(self):
        super().__init__()
        # define word embedding module:
        embedding_dim = 512
        hidden_dim = 1000
        self.embedding = nn.Embedding(num_embeddings=50257,embedding_dim=embedding_dim, dtype=torch.float64)
        # define lstm module:
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=False, dtype=torch.float64, batch_first=True)
        self.linear = nn.Linear(hidden_dim,1, dtype=torch.float64)

    def forward(self,x):
        embedded_x = self.embedding(x)

        lstm_outputs, _ = self.lstm(embedded_x.to(torch.float64))
        return self.linear(lstm_outputs[:,-1])
        pass

class StoryJudger(nn.Module):
    """
    Try to make actual story judger using encoder output, LSTM, linear layer? 
    """
    def __init__(self, encoder): # pass in pre-trained encoder
        super().__init__()
        # lstm + linear layer
        self.encoder = encoder
        paragraph_dim = 768
        hidden_dim = 1000
        self.lstm = nn.LSTM(paragraph_dim, hidden_dim, num_layers=2, bidirectional=True, dtype=torch.float64, batch_first=True)
        self.linear = nn.Linear(2*hidden_dim, 1, dtype=torch.float64)


    def forward(self, x):
        paragraph_embeddings = self.encoder(x)
        pre_output = self.lstm(paragraph_embeddings)
        return self.linear(pre_output[:, -1])

