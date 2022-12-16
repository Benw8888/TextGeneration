import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import DistilBertTokenizerFast, GPT2TokenizerFast, AutoTokenizer
# Implement lstm encoder
from transformers import GPT2TokenizerFast, DistilBertModel
import os
from pkg_resources import resource_filename

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Model,
    PreTrainedTokenizerFast,
)


STATIC_PATH = resource_filename(__name__, "aitextgen/static")

from transformers import AutoModel


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

    def forward(self,input_ids=None, length=None, **kwargs):
        # handle list of chunk inputs:
        embedded_x = self.embedding(input_ids)
        lstm_outputs, _ = self.lstm(embedded_x.to(torch.float64))
        print("length: ", length)
        sliced_outputs = torch.zeros(lstm_outputs[:,-1].shape, dtype=torch.float64)
        for i in range(lstm_outputs.shape[0]): # cut off at length
            sliced_outputs[i,:] = lstm_outputs[i,length[i]-1]
        print("sliced out shape: ",sliced_outputs.shape)
        return self.linear(sliced_outputs.to(torch.device("cuda"))) # length ensures we dont continue to pad tokens

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass

class EncoderGPT(pl.LightningModule):
    """
    Try to make an transformer encoder, using distilbert
    """
    def __init__(self, level="paragraph"):
        super().__init__()

        #self.gpt2_tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(STATIC_PATH, "gpt2_vocab.json"),
        #                              merges_file=os.path.join(STATIC_PATH, "gpt2_merges.txt"), padding_side="right")
        #self.gpt2_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})


        # TODO: recreate your own that's expanded, don't use pretrained
        self.config, kwargs = AutoConfig.from_pretrained(
                "gpt2",
                return_unused_kwargs=True,
                trust_remote_code=False,
                cache_dir="aitextgen",
            )
        self.model = GPT2Model(self.config)


        # define word embedding module:
        embedding_dim = 512
        self.hidden_dim = 768
        self.output_dim = 768
        #self.embedding = nn.Embedding(num_embeddings=50257,embedding_dim=embedding_dim, dtype=torch.float64)
        # define lstm module:
        linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        linear3 = nn.Linear(self.output_dim, self.output_dim)

        self.linear_layers = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2,
            nn.ReLU(),
            linear3,
            nn.ReLU(),
        ).to(torch.device("cuda"))

    def forward(self,input_ids=None, length=None, **kwargs):
        batch_size = input_ids.shape[0]
        # first create attention mask # not necessary

        # then run through gpt model

        gpt_output = self.model(input_ids=input_ids).last_hidden_state

        # then extract hidden outputs

        #model_outputs

        # finally average and pass through some linear layers

        assert self.hidden_dim == gpt_output.shape[2]

        average_model_outputs = torch.empty((batch_size,self.hidden_dim)).to(torch.device("cuda"))
        for batch in range(batch_size):
            average_model_outputs[batch] = torch.mean(gpt_output[batch, :length[batch], :], 0)

        average_model_outputs = average_model_outputs.to(torch.device("cuda"))

        # apply linear layers:

        output = self.linear_layers(average_model_outputs)

        return output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass

class EncoderBert(pl.LightningModule):
    """
    Try to make an transformer encoder, using distilbert
    """
    def __init__(self, level="paragraph"):
        super().__init__()

        self.bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.gpt2_tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(STATIC_PATH, "gpt2_vocab.json"),
                                      merges_file=os.path.join(STATIC_PATH, "gpt2_merges.txt"), padding_side="right")
        self.gpt2_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})


        # TODO: recreate your own that's expanded, don't use pretrained
        self.model = AutoModel.from_pretrained("distilbert-base-uncased", num_labels=2)


        # define word embedding module:
        embedding_dim = 512
        hidden_dim = 768
        output_dim = 768
        self.embedding = nn.Embedding(num_embeddings=50257,embedding_dim=embedding_dim, dtype=torch.float64)
        # define lstm module:
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, bidirectional=True, dtype=torch.float64, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2,output_dim, dtype=torch.float64)

    def forward(self,input_ids=None, length=None, **kwargs):
        # First decode gpt2 tokens back to strings
        print(input_ids.shape)
        raw_strings = np.stack([self.gpt2_tokenizer.decode(input_ids[i,:]) for i in range(input_ids.shape[0])], axis=0)
        print("raw strings shape: ",raw_strings.shape)

        # Then encode distilbert tokenizer
        print("first tokenized: ",self.bert_tokenizer(raw_strings[0]))
        bert_tokens = np.stack([self.bert_tokenizer(raw_strings[i]) for i in range(len(raw_strings))], axis=0)
        print("bert tok shape: ",bert_tokens.shape)
        print(bert_tokens)

        # then run distilbert

        return self.model(bert_tokens)
        # then take average of unmasked output vectors




        # handle list of chunk inputs:
        embedded_x = self.embedding(input_ids)
        lstm_outputs, _ = self.lstm(embedded_x.to(torch.float64))
        print("length: ", length)
        sliced_outputs = torch.zeros(lstm_outputs[:,-1].shape, dtype=torch.float64)
        for i in range(lstm_outputs.shape[0]): # cut off at length
            sliced_outputs[i,:] = lstm_outputs[i,length[i]-1]
        print("sliced out shape: ",sliced_outputs.shape)
        return self.linear(sliced_outputs.to(torch.device("cuda"))) # length ensures we dont continue to pad tokens

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass






