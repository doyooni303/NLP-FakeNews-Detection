from transformers import AutoConfig, BertModel, BertPreTrainedModel
from transformers import PreTrainedModel, RobertaPreTrainedModel
import torch.nn as nn
from transformers import AutoModel

from .registry import register_model

import logging
_logger = logging.getLogger('train')

class BERT_LSTM_m2o(BertPreTrainedModel):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int,
                 **kwargs):
        super().__init__(config)

        self.roberta = True if 'roberta' in pretrained_name else False

        self.bert = AutoModel.from_pretrained(pretrained_name, config=config)

        self.lstm = nn.LSTM(input_size = config.hidden_size, 
                            batch_first = True,
                            **kwargs
                            # hidden_size = lstm_hid,
                            # num_layers  = num_layers,
                            )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(nn.Linear(self.lstm.hidden_size, 128),
                        nn.BatchNorm1d(128),
                        nn.Dropout(0.6),
                        nn.Linear(128, 32),
                        nn.Linear(32, 2)
                        )        


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        hidden = None,
        cell = None
    ):

        if self.roberta:

            outputs = self.bert(
                input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = None,
                position_ids         = None,
                head_mask            = None,
                inputs_embeds        = None,
                output_attentions    = output_attentions,
                output_hidden_states = None,
            )
        else:
            outputs = self.bert(
                input_ids,
                attention_mask       = attention_mask,
                token_type_ids       = token_type_ids,
                position_ids         = None,
                head_mask            = None,
                inputs_embeds        = None,
                output_attentions    = output_attentions,
                output_hidden_states = None,
            )
            
        pooled_output = outputs[0]

        pooled_output, _ = self.lstm(
                                    pooled_output, 
                                    (hidden, cell)
                                )

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output[:,-1,:].reshape(-1, self.lstm.hidden_size))

        if output_attentions:
            return logits, outputs[-1]
        else:
            return logits

@register_model
def bert_lstm_m2o(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = BERT_LSTM_m2o(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes'],
        **kwargs
    )


    return model
