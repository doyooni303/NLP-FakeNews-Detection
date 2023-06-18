from transformers import AutoConfig, BertModel, BertPreTrainedModel, AutoModel
import torch
import torch.nn as nn
import pdb

from .registry import register_model

import logging
_logger = logging.getLogger('train')

class RoBERTa_DualBERT(BertPreTrainedModel):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int):
        super().__init__(config)

        self.roberta = AutoModel.from_pretrained(pretrained_name, config=config)
        self.linear = nn.Linear(2*config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)


    def forward(
        self,
        main=None,
        ctg=None,
    ):

        # get [cls] vectors
        main_outputs = self.roberta(
            main['input_ids'],
            attention_mask       = main['attention_mask'],
            token_type_ids       = None,
            position_ids         = None,
            head_mask            = None,
            inputs_embeds        = None,
            output_attentions    = None,
            output_hidden_states = None,
            )
        cls_main = main_outputs.pooler_output # [CLS] token

        ctg_outputs = self.roberta(
            ctg['input_ids'],
            attention_mask       = ctg['attention_mask'],
            token_type_ids       = None,
            position_ids         = None,
            head_mask            = None,
            inputs_embeds        = None,
            output_attentions    = None,
            output_hidden_states = None,
            )
        cls_ctg = ctg_outputs.pooler_output # [CLS] token

        # concatenate two [cls] vectors
        cls_concat = torch.cat([cls_main,cls_ctg],dim=1)

        # passing linear layer        
        output = self.linear(cls_concat)
        pooled_output = self.dropout(output)
        logits = self.classifier(pooled_output)

        return logits

@register_model
def RoBERTa_dualbert(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = RoBERTa_DualBERT(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes']
    )
    # model.resize_token_embeddings(len(self.tokenizer))

    return model