import numpy as np
import torch
import torch.nn as nn

# Implement story judger


class StoryJudger(nn.Module):
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
