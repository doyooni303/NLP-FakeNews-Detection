from transformers import AutoConfig, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
from .registry import register_model

class Multimodal(BertPreTrainedModel):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int, cat_num: int = 175):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size+cat_num, 128), #cat_num=175
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(128, 128),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(128, num_classes)
                                        )

    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        length_of_tokens=None,
        category=None
        ):

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


        pooled_output = outputs[1]

        mid_output = torch.cat([pooled_output,category],dim=1)
        logits = self.classifier(mid_output)
        
        if output_attentions:
            return logits, outputs[-1]
        else:
            return logits

@register_model

def Multimodal_net(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = Multimodal(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes'],
        **kwargs
    )

    return model