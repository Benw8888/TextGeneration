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

    def forward(self, data):
        # data is batch by paragraphs by token ids, assume batch=1

        # encoder needs pbatch by token ids, so we reshape: (treat each paragraph in a story as a separate batch)
        encoder_input = data[0,:,:]
        paragraph_embeddings = self.encoder(encoder_input)

        # re add the 1 dummy batch size dimension
        batched_paragraph_embeddings = paragraph_embeddings[None,:,:]

        pre_output, _ = self.lstm(batched_paragraph_embeddings)
        return self.linear(pre_output[:, -1])

    def save_pretrained(self, target_folder=None):
        """Saves the model into the specified directory."""
        print("SAVING BASED ON: ", target_folder)
        torch.save(self.state_dict(), target_folder)

class StoryJudger2(nn.Module):
    """
    Try to make a story judger which combines local word-level evaluations with paragraph embedding evaluations?
    """
    def __init__(self, encoder): # pass in pre-trained encoer
        super().__init__()
        self.encoder = encoder
        embedding_dim = 512
        word_hidden_dim = 1000
        self.embedding = nn.Embedding(num_embeddings=50257, embedding_dim=embedding_dim, dtype = torch.float64)
        # first lstm for word embeddings
        self.lstm1 = nn.LSTM(embedding_dim, word_hidden_dim, num_layers=2, bidirectional=False, dtype=torch.float64, batch_first=True)
        self.linear1 = nn.Linear(word_hidden_dim, 1, dtype=torch.float64)
        # paragraph embedding evaluation modules
        # lstm + linear layer
        self.encoder = encoder
        paragraph_dim = 768
        paragraph_hidden_dim = 1000
        self.lstm2 = nn.LSTM(paragraph_dim, paragraph_hidden_dim, num_layers=2, bidirectional=True, dtype=torch.float64, batch_first=True)
        self.linear2 = nn.Linear(2*paragraph_hidden_dim, 1, dtype=torch.float64)
        self.comb_linear = nn.Linear(2, 1, dtype=torch.float64) # final linear layer to combine the word and paragraph scores

    def forward(self, x):
        paragraph_embeddings = self.encoder(x)
        paragraph_outputs, _ = self.lstm2(paragraph_embeddings)
        paragraph_scores = self.linear2(paragraph_outputs[:, -1])
        embedded_x = self.embedding(x)
        word_lstm_outputs, _ = self.lstm1(embedded_x.to(torch.float64))
        word_scores = self.linear1(word_lstm_outputs[:, -1])
        combo_scores = torch.cat([paragraph_scores, word_scores], -1)
        return self.comb_linear(combo_scores)

