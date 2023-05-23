from transformers import AutoConfig, BertModel, BertPreTrainedModel
import torch.nn as nn
from .registry import register_model
import pdb
import torch
import logging
import numpy as np
_logger = logging.getLogger('train')


# 원래 설정: cat_num 175, nn.Linear(config.hidden_size+cat_num+2, 128)
# one cat설정: cat_num 7, nn.Linear(config.hidden_size+cat_num+1, 128) 

class BERT(BertPreTrainedModel):
    def __init__(self, pretrained_name: str, config: dict, num_classes: int, cat_num: int = 0):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(pretrained_name, config=config)
        # self.ptfile = torch.load('/workspace/Fake-News-Detection-Dataset/part1_siamese/normal_bert.pt')
        # self.bert.load_state_dict(self.ptfile)
        self.cat_num = cat_num
        self.number_importance_per_cat = nn.Parameter(torch.ones(7, 15))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size+cat_num+15, 128), #cat_num=175
                                        nn.LeakyReLU(0.1),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(128, 128),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(128, 2)
                                        )
        
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        length_of_tokens=None,
        category=None,
        # continuous=None,
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
        category = category[:, :7]
        # number_importance_per_cat = torch.randn(7, 15, requires_grad=True).cuda()
        
        this_time = torch.matmul(category, self.number_importance_per_cat)
        
        pooled_output = outputs[1].reshape(-1,self.config.hidden_size)
        for i, representations in enumerate(outputs[0]):
            
            if i==0:
                title_rep = torch.mean(representations[1:length_of_tokens[i][0].item()+1], dim=0, keepdim=True).unsqueeze(0)
                # batch_title = torch.vstack([batch_title, title_rep])
                start = length_of_tokens[i][0].item()+1+1 # cls + 제목 + sep
                
                for idx, leng in enumerate(length_of_tokens[i][1:]):
                    
                    if idx==0:
                        
                        leng = leng.item()
                        if (leng==0 or start>=512):                             
                            break
                        
                        end = start + leng
                        if end >= 512:
                            end = 512
                            content_rep = torch.mean(representations[start:end], dim=0, keepdim=True)
                            # batch_contents = torch.vstack([batch_contents, content_rep])
                            
                        
                        else:
                            content_rep = torch.mean(representations[start:end], dim=0, keepdim=True)
                            # batch_contents = torch.vstack([batch_contents, content_rep])
                            
                        start = end
                    
                        
                    else: 
                        
                        leng = leng.item()
                        if (leng==0 or start>=512):                             
                            break
                        
                        end = start + leng
                        if end >= 512:
                            end = 512
                            content_rep = torch.vstack([content_rep, torch.mean(representations[start:end], dim=0, keepdim=True)])

                            
                        else:
                            content_rep = torch.vstack([content_rep, torch.mean(representations[start:end], dim=0, keepdim=True)])
                            
                        start = end
                
                if content_rep.shape[0] >= 15:
                    content_rep = content_rep[0:15]
                else:
                    cnt = 15 - content_rep.shape[0]
                    content_rep = torch.vstack([content_rep, torch.zeros(cnt, content_rep.shape[1]).cuda()])
                    
                content_representations = content_rep.unsqueeze(0)
            
            
            else:
                title_rep = torch.vstack([title_rep, torch.mean(representations[1:length_of_tokens[i][0].item()+1], dim=0, keepdim=True).unsqueeze(0)])
            
                start = length_of_tokens[i][0].item()+1+1 # cls + 제목 + sep
                
                for idx, leng in enumerate(length_of_tokens[i][1:]):
                    
                    if idx==0:
                        
                        leng = leng.item()
                        if (leng==0 or start>=512):                             
                            break
                        
                        end = start + leng
                        if end >= 512:
                            end = 512
                            content_rep = torch.mean(representations[start:end], dim=0, keepdim=True)
                            # batch_contents = torch.vstack([batch_contents, content_rep])
                            
                        
                        else:
                            content_rep = torch.mean(representations[start:end], dim=0, keepdim=True)
                            # batch_contents = torch.vstack([batch_contents, content_rep])
                            
                        start = end
                    
                        
                    else: 
                        
                        leng = leng.item()
                        if (leng==0 or start>=512):                             
                            break
                        
                        end = start + leng
                        if end >= 512:
                            end = 512
                            content_rep = torch.vstack([content_rep, torch.mean(representations[start:end], dim=0, keepdim=True)])

                            
                        else:
                            content_rep = torch.vstack([content_rep, torch.mean(representations[start:end], dim=0, keepdim=True)])
                            
                        start = end
                
                if content_rep.shape[0] >= 15:
                    content_rep = content_rep[0:15]
                else:
                    cnt = 15 - content_rep.shape[0]
                    content_rep = torch.vstack([content_rep, torch.zeros(cnt, content_rep.shape[1]).cuda()])
                    
                content_representations = torch.vstack([content_representations, content_rep.unsqueeze(0)])
        
        
        
        sims = torch.nn.functional.cosine_similarity(title_rep, content_representations, dim=2)
        
        sims_importance = torch.mul(sims, this_time)
        
        # mid_output = torch.cat([pooled_output,category,continuous, sims], dim=1)
        mid_output = torch.cat([pooled_output, sims_importance], dim=1)
        
        logits = self.classifier(mid_output)

        if output_attentions:
            return logits, outputs[-1]
        else:
            return logits 

@register_model
def bert(hparams: dict, **kwargs):
    model_config = AutoConfig.from_pretrained(hparams['pretrained_name'])
    model = BERT(
        pretrained_name = hparams['pretrained_name'], 
        config          = model_config,
        num_classes     = hparams['num_classes']
    )

    return model
