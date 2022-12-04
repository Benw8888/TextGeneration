import numpy as np
import torch
import torch.nn as nn
import os
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)
from transformers.models.gpt2.convert_gpt2_original_tf_checkpoint_to_pytorch import (
    convert_gpt2_checkpoint_to_pytorch,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


# Implement gpt2 decoder conditioned on paragraph embedding

# config, kwargs = AutoConfig.from_pretrained(
#                 "gpt2",
#                 return_unused_kwargs=True,
#                 trust_remote_code=False,
#                 cache_dir="aitextgen",
#             )
#print(config)
#print("config cross attention", config.add_cross_attention)
#print("config inner",config.n_inner )
##print(config.hidden_size)  # hidden size is 768


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]

class Decoder(pl.LightningModule):
    """
    Try to make an LSTM encoder
    """
    def __init__(self, embedding_size = 768, file_path=None,level="paragraph"):
        super().__init__()
        cache_dir = "aitextgen"
        #config = os.path.join(cache_dir, f"config_{tf_gpt2}.json")
        if file_path is not None:
            self.model = self.load_model(file_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "gpt2", cache_dir=cache_dir
            )

        self.config, self.kwargs = AutoConfig.from_pretrained(
            "gpt2",
            return_unused_kwargs=True,
            trust_remote_code=False,
            cache_dir="aitextgen",
        )

        self.embedding_size = embedding_size  # Size of chunk embedding. MAKE SURE DIVISIBLE BY 12

        # still of type GPT2LMHeadModel
        
        print("using gpt2 model of type: ",type(self.model).__name__)

        self.modify_gpt2()

        # create custom modified GPT2LMHeadModel

    def load_model(self, model_folder):
        # A folder is provided containing pytorch_model.bin and config.json
        assert os.path.exists(
            os.path.join(model_folder, "pytorch_model.bin")
        ), f"There is no pytorch_model.bin in /{model_folder}."
        assert os.path.exists(
            os.path.join(model_folder, "config.json")
        ), f"There is no config.json in /{model_folder}."

        logger.info(
            f"Loading model from provided weights and config in /{model_folder}."
        )
        return AutoModelForCausalLM.from_pretrained(
            model_folder, local_files_only=True
        )

    def modify_gpt2(self):
        """
        Modifies gpt2 to take in expanded input that is concatenated by paragraph embedding
        """
        # Now we need to update block.attn and block.ln_2 and block.mlp; also might have to deal with the layer_past stuff (may cause problems?)
        # un-expand in the block.mlp
        #self.block.forward = self.overwrite_block_forward

        # First, expand weights
        # expand hidden size from 768 to 768+self.embedding_size
        self.block_list = self.model.transformer.h
        self.block = self.block_list[0]
        #self.block.ln_1 #= nn.LayerNorm(768 + self.embedding_size, eps=self.config.layer_norm_epsilon)

        ln_1 = self.block.ln_1
        ln_1.normalized_shape = (ln_1.normalized_shape[0]+self.embedding_size,)
        ln_1.weight = nn.Parameter(torch.concat([ln_1.weight,torch.ones(self.embedding_size)],dim=0))
        ln_1.bias = nn.Parameter(torch.concat([ln_1.bias, torch.zeros(self.embedding_size)], dim=0))

        #self.block.ln_2 #= nn.LayerNorm(768 + self.embedding_size, eps=self.config.layer_norm_epsilon)

        ln_2 = self.block.ln_2
        ln_2.normalized_shape = (ln_2.normalized_shape[0]+self.embedding_size,)
        ln_2.weight = nn.Parameter(torch.concat([ln_2.weight, torch.ones(self.embedding_size)], dim=0))
        ln_2.bias = nn.Parameter(torch.concat([ln_2.bias, torch.zeros(self.embedding_size)], dim=0))
        
        
        # CHANGE ATTENTION

        # NOTE: make sure attn mask is 0 or -10000
        attn = self.block.attn

        c_attn = attn.c_attn
        columns = [] # the columns of c_attn
        bias_chunks = []

        # change attn weight and bias
        # EXPAND COLUMNS
        for attn_type in range(3): # query, key, value
            for attn_head in range(12):
                # column range from attn_type*768+attn_head * 64
                attn_head_start_column = attn_type*768+attn_head * 64
                columns.append(c_attn.weight[:, attn_head_start_column : attn_head_start_column + int(self.embedding_size/12)])  # copy parts of c_attn
                columns.append(torch.zeros(768, int(self.embedding_size/12)))

                bias_chunks.append(c_attn.bias[attn_head_start_column: attn_head_start_column + int(self.embedding_size/12)])
                bias_chunks.append(torch.zeros(int(self.embedding_size/12)))

        c_attn.weight = nn.Parameter(torch.concat(columns, dim=1))
        c_attn.bias = nn.Parameter(torch.concat(bias_chunks, dim=0))

        # EXPAND ROWS
        c_attn.weight = nn.Parameter(torch.concat([c_attn.weight,torch.zeros(self.embedding_size, 3*768+3*self.embedding_size)], dim=0))

        c_attn.nf = 3*(self.config.hidden_size+self.embedding_size)

        attn.embed_dim = self.config.hidden_size+self.embedding_size
        attn.head_dim = attn.embed_dim // attn.num_heads
        attn.split_size = attn.embed_dim

        # Attention Proj
        attn_proj = self.block.attn.c_proj #= Conv1D(self.embed_dim, self.embed_dim)
        attn_proj.weight = nn.Parameter(torch.concat([attn_proj.weight, torch.zeros((768,self.embedding_size))],dim=1))
        attn_proj.weight = nn.Parameter(torch.concat([attn_proj.weight, torch.zeros((self.embedding_size, 768+self.embedding_size))], dim=0))
        attn_proj.bias = nn.Parameter(torch.concat([attn_proj.bias, torch.zeros(self.embedding_size)], dim=0))
        attn_proj.nf = attn_proj.nf + self.embedding_size



        
        
        # CHANGE MLP
        hidden_size = 768
        mlp = self.block.mlp
        current_c_fc_weight = mlp.c_fc.weight
        #embed x 4*hidden_size
        first_zeros_c_fc = torch.zeros(hidden_size, 4*self.embedding_size)
        current_c_fc_weight = torch.cat([current_c_fc_weight,first_zeros_c_fc], dim=1)
        #now hidden x (4*hidden_size + 4*embedding_size)
        second_zeros_c_fc = torch.zeros(self.embedding_size, 4*self.embedding_size+4*hidden_size)
        current_c_fc_weight = torch.cat([current_c_fc_weight,second_zeros_c_fc], dim = 0)
        mlp.c_fc.weight = nn.Parameter(current_c_fc_weight)

        current_c_fc_bias = mlp.c_fc.bias
        zeros_c_fc_bias = torch.zeros(4*self.embedding_size)
        current_c_fc_bias = torch.cat([current_c_fc_bias, zeros_c_fc_bias], dim=0)
        mlp.c_fc.bias = nn.Parameter(current_c_fc_bias)

        mlp.c_fc.nf = mlp.c_fc.nf + 4*self.embedding_size

        #proj:
        current_c_proj_weight = mlp.c_proj.weight
        #4*hidden_size x embed 
        first_zeros_c_proj = torch.zeros(4*self.embedding_size, hidden_size)
        current_c_proj_weight = torch.cat([current_c_proj_weight,first_zeros_c_proj], dim=0)
        #now (4*hidden_size + 4*embedding_size) x hidden
        mlp.c_proj.weight = nn.Parameter(current_c_proj_weight)

        # Now, overwrite forward methods
        self.block.attn.forward = self.overwrite_attn_forward
        self.block.forward = self.overwrite_block_forward
        self.model.transformer.forward = self.overwrite_gpt2_forward
        self.model.forward = self.overwrite_gpt2lmhead_forward

    def overwrite_attn_forward(
        decoder,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        self = decoder.block.attn
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            print("hidden states shape: ", hidden_states.shape)
            print("c attn WEIGHT shape: ", self.c_attn.weight.shape)
            print("c attn BIAS shape: ", self.c_attn.bias.shape)
            x = hidden_states
            print("HIDDEN STATES VIEW SHAPE: ",x.view(-1, x.size(-1)).shape)
            print("qkv c_attn: ", self.c_attn(hidden_states).shape)
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

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


    def overwrite_gpt2_forward(
        decoder,
        paragraph_embedding = None,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        self = decoder.model.transformer

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        print("INPUTs EMBEDS SHAPE: ", inputs_embeds.shape)
        output_shape_compressed= input_shape + (inputs_embeds.size(-1),)

        # CONCATENATE PARAGRAPH EMBEDDINGS TO INPUTS
        necessary_repeating = inputs_embeds.shape[1]
        hidden_states = torch.cat((inputs_embeds + position_embeds,
                                   paragraph_embedding.unsqueeze(1).repeat(1, necessary_repeating, 1)), dim=-1)


        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape_expanded = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if i==0:
                output_shape = output_shape_expanded
            else:
                output_shape = output_shape_compressed

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def overwrite_gpt2lmhead_forward(
        decoder,
        input_ids: Optional[torch.LongTensor] = None,
        conditioned_on = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        self = decoder.model
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            paragraph_embedding=conditioned_on,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def forward(self,embedding, x):
        #return self.model(input_ids= x, labels=x)
        return self.model(input_ids= x, labels=x, conditioned_on=embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters,lr=0.001)
        return optimizer

    def training_step(self, batch_inp, step_id):
        pass
