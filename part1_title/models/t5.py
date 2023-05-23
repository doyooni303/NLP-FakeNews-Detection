from collections import OrderedDict

from transformers import (
    T5Model,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoConfig,
)

import torch
import torch.nn as nn
import pdb
from .registry import register_model

import logging

_logger = logging.getLogger("train")


class FeedForwardLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout_rate: float = 0.5
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        self._init_weights(self.linear.weight)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer = self._set_layer()

    def _set_layer(self):
        return nn.Sequential(
            self.linear,
            self.dropout,
            self.relu,
            self.softmax,
        )

    def _init_weights(self, weight):
        nn.init.xavier_normal_(weight)

    def forward(self, input):
        output = self.layer(input)
        return output


class T5EncNet(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        config: dict,
        num_classes: int,
        pooler: str,
        T5Gen_path: str = None,
    ) -> None:
        super().__init__()

        self.T5Gen = T5Gen(pretrained_name)
        if T5Gen_path is not None:
            self.T5Gen.load_state_dict(torch.load(T5Gen_path))

        self.encoder = self.T5Gen.model.get_encoder()
        self.fflayer = FeedForwardLayer(
            in_features=config.d_model,
            out_features=num_classes,
        )

        self.pooler = pooler

    def forward(self, input_ids, attention_mask):
        
        output = self.encoder(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze(),
            output_hidden_states = True,
        ).last_hidden_state

        # To make tensor size as (batch_size, dim)
        pooled_output = self.pooling(output, self.pooler)
        final_output = self.fflayer(pooled_output)

        return final_output

    def pooling(self, x, pooler: str):
        
        if pooler == "last":
            pooled_output = x[:, -1, :]
        elif pooler == "mean":
            pooled_output = torch.mean(x,1).values
        elif pooler == "max":
            pooled_output = torch.max(x,1).values

        return pooled_output

class T5Gen(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
    ) -> None:
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_name)

    def forward(self, inputs, labels):
        return self.model(**inputs, labels=labels)
    
    def generate(self,inputs):
        return self.model.generate(**inputs)


@register_model
def t5gen(hparams: dict, **kwargs):
    model = T5Gen(
        pretrained_name = hparams['pretrained_name'], 
    )
    return model

@register_model
def t5encnet(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = T5EncNet(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes'],
        pooler          = hparams['pooler'],
        T5Gen_path      = hparams['T5Gen_path'],
    )

    return model