from .build_dataset import FakeDataset
from glob import glob
import os
import json
import torch

import logging
from typing import Union

_logger = logging.getLogger('train')

class T5GenDataset(FakeDataset):
    def __init__(self, tokenizer, max_source_length: int, max_target_length: int):
        super(T5GenDataset, self).__init__(tokenizer = tokenizer)
        
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.vocab = self.tokenizer.vocab

        # special token index
        self.pad_idx = self.tokenizer.pad_token_id # <pad>:0
        self.eos_idx = self.tokenizer.eos_token_id # </s>:1

       
    def input_transform(self, content: str, subtitle: str=None, task_prefix:str = "낚시성 뉴스 기사 탐지 및 제목 생성: ") -> dict:
        if subtitle!="":
            input_sequences = task_prefix + subtitle + content
        else:
            input_sequences = task_prefix + content
        encoder_encoding = self.tokenizer(
            input_sequences,
            padding = "max_length",
            max_length = self.max_source_length,
            truncation = True,
            return_tensors = "pt",
        )

        
        doc = dict(
            input_ids = encoder_encoding['input_ids'],
            attention_mask = encoder_encoding['attention_mask'],
        )

        return doc

    def label_transform(self, title: str) -> torch.tensor:

        label = self.tokenizer(
            title,
            padding="max_length",
            max_length = self.max_target_length,
            truncation = True,
            return_tensors = "pt",
        ).input_ids

        return label



    def __getitem__(self, i: int) -> Union[dict, int]:
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_info = self.data[self.data_info[i]]
            
            # label
            label = self.label_transform(
                title = news_info['sourceDataInfo']['newsTitle']
            )
        
            # transform and padding
            doc = self.input_transform(
                content = news_info['sourceDataInfo']['newsContent'],
                subtitle = news_info['sourceDataInfo']['newsSubTitle']
            )

            return doc, label
    
    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)
        
    def transform(self):
        pass
    
    def padding(self):
        pass



class T5EncNetDataset(FakeDataset):
    def __init__(self, tokenizer, max_source_length: int,):
        super(T5EncNetDataset,self).__init__(tokenizer = tokenizer)
        self.max_source_length = max_source_length
        self.vocab = self.tokenizer.vocab

        # special token index
        self.pad_idx = self.tokenizer.pad_token_id # <pad>:0
        self.eos_idx = self.tokenizer.eos_token_id # </s>:1
        
        
    
    def transform(self, title: str, content: list, subtitle: str=None, task_prefix:str = "낚시성 뉴스 기사 탐지 및 제목 생성: ") -> dict:
        if subtitle!="":
            input_sequences = task_prefix + title + subtitle + content
        else:
            input_sequences = task_prefix + title + content
        
        encoder_encoding = self.tokenizer(
            input_sequences,
            padding = "max_length",
            max_length = self.max_source_length,
            truncation = True,
            return_tensors = "pt",
        )

        doc = dict(
            input_ids = encoder_encoding['input_ids'],
            attention_mask = encoder_encoding['attention_mask'],
        )


        return doc


    def __getitem__(self, i: int) -> Union[dict, int]:
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]

            return doc, label
        
        else:
            news_info = self.data[self.data_info[i]]
        
            # label
            label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
        
            # transform
            doc = self.transform(
                title = news_info['labeledDataInfo']['newTitle'], 
                content = news_info['sourceDataInfo']['newsContent'],
                subtitle = news_info['sourceDataInfo']['newsSubTitle']
            )

            return doc, label
    
    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)
        
    def padding(self):
        pass
    