from transformers import AutoConfig, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from .registry import register_model

import logging
_logger = logging.getLogger('train')

class Siamese_BERT(BertPreTrainedModel):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int, pred_dim: int):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.predictor = nn.Sequential(nn.Linear(config.hidden_size, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, config.hidden_size)) # output layer
        
        self.classifier = nn.Linear(config.hidden_size*2, num_classes)


    def forward(self, title, contents):

        title_outputs = self.bert(
            input_ids            = title['input_ids'],
            attention_mask       = title['attention_mask'],
            token_type_ids       = title['token_type_ids'],
            # position_ids         = None,
            # head_mask            = None,
            # inputs_embeds        = None,
            # output_attentions    = None,
            # output_hidden_states = None,
        )
        
        contents_outputs = self.bert(
            input_ids            = contents['input_ids'],
            attention_mask       = contents['attention_mask'],
            token_type_ids       = contents['token_type_ids'],
            # position_ids         = None,
            # head_mask            = None,
            # inputs_embeds        = None,
            # output_attentions    = None,
            # output_hidden_states = None,
        )
            
        title_z = title_outputs[1].detach()
        title_p = self.predictor(title_z)
        
        contents_z = contents_outputs[1]

        final_input = torch.cat([title_p, contents_z], dim=1)
        pooled_output = self.dropout(final_input)
        logits = self.classifier(pooled_output)


        return logits

@register_model
def siamese_bert(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = Siamese_BERT(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes']
        pred_dim        = 192
    )

    return model
