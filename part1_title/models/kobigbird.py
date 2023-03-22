from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig, BertModel, BertPreTrainedModel
import torch.nn as nn

from .registry import register_model

import logging
_logger = logging.getLogger('train')

class KOBIGBIRD(nn.Module):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int):
        # super(KOBIGBIRD, self).__init__(config)
        super(KOBIGBIRD, self).__init__()

        self.kobigbird = AutoModel.from_pretrained(pretrained_name, config=config)  
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
    ):

        outputs = self.kobigbird(
            input_ids,
            attention_mask       = attention_mask,
            token_type_ids       = token_type_ids,
            position_ids         = None,
            head_mask            = None,
            inputs_embeds        = None,
            output_attentions    = output_attentions,
            output_hidden_states = None,
        )
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_attentions:
            return logits, outputs[-1]
        else:
            return logits

@register_model
def kobigbird(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = KOBIGBIRD(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes']
    )

    return model
