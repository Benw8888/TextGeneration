import math
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Block, GPT2Attention, GPT2MLP
from transformers import AutoConfig

from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import Encoder
import Decoder


class TextGenerator(nn.Module):
    def __init__(self, generator_file_path, decoder_file_path, size_input=30, device=None):
        super().__init__()
        self.config, self.kwargs = AutoConfig.from_pretrained(
            "gpt2",
            return_unused_kwargs=True,
            trust_remote_code=False,
            cache_dir="aitextgen",
        )
        self.device_to_use = device

        self.decoder = Decoder.Decoder()
        self.decoder.load_state_dict(torch.load(decoder_file_path))

        self.generator = ChunkGPT(self.config, size_input)
        self.generator.load_state_dict(torch.load(generator_file_path))

        # INFO
        self.size_input = size_input # max size of input for generator

    def forward(self, x=None):
        # Assume input of shape 1 x num_chunks x 768
        if x is None:
            shifted_input = self.create_start_vectors(1)
        else:
            shifted_input = torch.concat(
                [self.create_start_vectors(1).to(self.device_to_use), x[:, :, :]],  # no need to cut off end of x this time
                dim=1)
        assert shifted_input.shape[1]<=self.size_input, f"size of text generator input greater than allowed, got: {shifted_input.shape[1]-1}"

        chunk_vectors = self.generate_all_chunks(shifted_input)
        print("chunk vectors: ",chunk_vectors)

        # now expand each chunk vector into tokens/words

        text = self.decode_chunks(chunk_vectors)



        pass

    def create_start_vectors(self, batch_size):
        return torch.zeros((1, 1, 768))

    def decode_chunks(self):

        pass

    def generate_all_chunks(self, shifted_input):
        # do one step generation, return past value keys somehow
        length = shifted_input.shape[1]
        # need to get length to self.size_input; this includes the zero start vector
        outputs = self.generator.forward(inputs_embeds=shifted_input, use_cache=True)
        next_vector = outputs.last_hidden_state[:,-1:,:]
        past_key_values = outputs.past_key_values
        shifted_input = torch.concat(
            [shifted_input, next_vector],
            dim=1)
        length += 1

        while length < self.size_input:
            outputs = self.generator.forward(inputs_embeds=next_vector, past_key_values=past_key_values, use_cache=True)
            next_vector = outputs.last_hidden_state[:,-1:,:]
            print("last hidden state shape: ", outputs.last_hidden_state.shape)
            past_key_values = outputs.past_key_values
            shifted_input = torch.concat(
                [shifted_input, next_vector],
                dim=1)
            length +=1

        return shifted_input[:,1:,:]
        # for loop for the rest, feeding in (only one vector?) and past key values
        pass



class GeneratorTrainer(pl.LightningModule):
    """
    Model adapted by Eric from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """

    # Constructor
    def __init__(self, dataset, hparams, encoder_file_path = None, generator_load_path=None,num_chunks=30,device=None):
        super().__init__()
        self.device_to_use = device

        self.dataset = dataset
        self.encoder = Encoder.Encoder()
        self.encoder.load_state_dict(torch.load(encoder_file_path))

        # INFO
        self.model_type = "ChunkGPT"

        # MODEL
        self.config, self.kwargs = AutoConfig.from_pretrained(
            "gpt2",
            return_unused_kwargs=True,
            trust_remote_code=False,
            cache_dir="aitextgen",
        )

        self.size_input = num_chunks
        self.config.n_positions = self.size_input

        self.model = ChunkGPT(self.config, self.size_input)
        if generator_load_path is not None:
            self.model.load_state_dict(torch.load(generator_load_path))

        self.loss_fn = torch.nn.functional.mse_loss

        self.save_hyperparameters(hparams)


    def forward(self,shifted_input):
        if isinstance(shifted_input, dict):
            print("INP IS DICT TO FORWARD")
            return self.model.forward(inputs_embeds=shifted_input['input_embeds'])
        return self.model.forward(inputs_embeds=shifted_input)



    def training_step(self, batch_inp, step_id):
        x = batch_inp
        if isinstance(x, dict):
            # data is batch by paragraphs by token ids, assume batch=1

            # encoder needs pbatch by token ids, so we reshape: (treat each paragraph in a story as a separate batch)
            with torch.no_grad():
                #print("x['input_ids'] :",x['input_ids'])
                #print("x inp shape: ",x['input_ids'].shape)
                encoder_input = x['input_ids'][0, :, :]
                paragraph_embeddings = self.encoder(encoder_input)

            # re add the 1 dummy batch size dimension
            batched_paragraph_embeddings = paragraph_embeddings[None, :, :]

            raw_input = batched_paragraph_embeddings
            x['labels'] = raw_input

            batch_size = raw_input.shape[0]
            shifted_input = torch.concat([self.create_start_vectors(batch_size).to(self.device_to_use), raw_input[:, :-1, :]], # middle :-1
                                         dim=1)  # shift input ids, add start
            #print("shifted input type: ", type(shifted_input))
            #print("shifted input shape: ", shifted_input.shape)
            #print("shifted input: ", shifted_input)
            x['input_embeds'] = shifted_input
            x['input_ids'] = None

            outputs = self.forward(shifted_input)
        else:
            # data is batch by paragraphs by token ids, assume batch=1

            # encoder needs pbatch by token ids, so we reshape: (treat each paragraph in a story as a separate batch)
            print("ERROR! Input to generator should be a dict")
            with torch.no_grad():
                encoder_input = batch_inp[0, :, :]
                paragraph_embeddings = self.encoder(encoder_input)

            # re add the 1 dummy batch size dimension
            batched_paragraph_embeddings = paragraph_embeddings[None, :, :]

            raw_input = batched_paragraph_embeddings
            labels = raw_input
            batch_size = raw_input.shape[0]
            shifted_input = torch.concat([self.create_start_vectors(batch_size), raw_input[:, :-1, :]],
                                         dim=1)  # shift input ids, add start
            outputs = self.forward(shifted_input)
        generated = outputs.last_hidden_state
        loss = self.loss_fn(raw_input.float(), generated)
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



    def create_start_vectors(self, batch_size, ):
        return torch.zeros((batch_size,1,768))
        pass

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int = -1) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)



class ChunkGPT(GPT2Model):
    def __init__(self, config, size_input=128):
        super().__init__(config)
        config.n_positions = size_input

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.block = self.h[0]

        self.block.forward = self.overwrite_block_forward

    def save(
        self,
        target_file
    ):
        torch.save(self.state_dict(), target_file)
        print("finished saving")


    def save_pretrained(
        self,
        target_file,
        **kwargs,
    ):
        torch.save(self.state_dict(), target_file)
        print("finished saving pretrained")

    def load(self, load_file):
        self.load_state_dict(torch.load(load_file))

    def overwrite_block_forward(self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,):

        hidden_states = hidden_states.float()
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)
        attn_outputs = self.block.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.block.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.block.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.block.ln_2(hidden_states)
        feed_forward_hidden_states = self.block.mlp(hidden_states)

        dimensions = feed_forward_hidden_states.size()
        assert(len(dimensions)==3)
        residual = residual[:dimensions[0], :dimensions[1], :dimensions[2]]

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


