from torch.utils.data import Dataset
import json 
import pandas as pd
import torch
import os
from glob import glob
import pdb

import logging
from typing import Union

_logger = logging.getLogger('train')

class FakeDataset(Dataset):
    def __init__(self, tokenizer, use_cat, name):
        # tokenizer
        self.tokenizer = tokenizer
        self.use_cat = use_cat
        self.name = name

    def load_dataset(self, data_dir, split, direct_dir: Union[None, str] = None, saved_data_path: bool = False, cat_keys = None):
        
        self.data_dir = data_dir
        data_info = glob(os.path.join(data_dir, split, '*/*/*'))
        self.data_info = data_info 
        
        if 'CAT_CONT_LEN' in self.name:
            self.cat_keys = cat_keys
        
        if direct_dir:
            exclude_info = glob(os.path.join(data_dir, split, 'Clickbait_Auto/*/*'))
            include_info = glob(os.path.join(direct_dir, split, '*/*/*'))
            data_info = list(set(data_info) - set(exclude_info))
            data_info = data_info + include_info

        setattr(self, 'saved_data_path', saved_data_path)
        
        if 'CAT_CONT_LEN' in self.name:
            if saved_data_path: 
                self.data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
                
            else:
                self.data = {}
                for filename in self.data_info:
                    f = json.load(open(filename,'r'))
                    self.data[filename] = f
                
                if not os.path.exists("./encoding_info.json"):
                    self.encoding_info=self.encoding_list()
                else:
                    with open("./encoding_info.json", "r") as encoding:
                        self.encoding_info=json.load(encoding)
        else:
            if saved_data_path:
                _logger.info('load saved data')
                data = torch.load(os.path.join(saved_data_path, f'{split}.pt'))
                
            else:
                _logger.info('load raw data')
                    
                data = {}
                for filename in data_info:
                    f = json.load(open(filename,'r'))
                    data[filename] = f
                    
            setattr(self, 'data_info', data_info)
            setattr(self, 'data', data)



    def transform(self):
        raise NotImplementedError

    def padding(self):
        raise NotImplementedError

    def __getitem__(self, i: int) -> Union[dict, int]:
        
        if self.saved_data_path:
            doc = {}
            for k in self.data['doc'].keys():
                doc[k] = self.data['doc'][k][i]

            label = self.data['label'][i]
            
            #! 
            if 'CAT_CONT_LEN' in self.name:
                length_of_tokens = self.data['length_of_tokens'][i]
                cat_tensor = self.data['cat_doc'][i]
                return doc, label, cat_tensor, length_of_tokens
            
            else:
                return doc, label 
        else:
            news_info = self.data[self.data_info[i]]
        
            # label
            label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
            
            
            #! 
            if self.use_cat:
                doc = self.transform(
                                    title = news_info['labeledDataInfo']['newTitle'], 
                                    text  = news_info['sourceDataInfo']['newsContent'].split('\n'),
                                    category = news_info['sourceDataInfo']['newsCategory'],
                                    subcategory = news_info['sourceDataInfo']['newsSubcategory']
                                    )
                    
                return doc, label
        
            else:
                news_info = self.data[self.data_info[i]]
                # label
                label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
                
                #!
                if 'CAT_CONT_LEN' in self.name:
                    length_of_tokens, doc = self.transform(
                        title = news_info['labeledDataInfo']['newTitle'], 
                        text  = news_info['sourceDataInfo']['newsContent'].split('\n')
                    )
                    
                    for idx, key in enumerate(self.cat_keys):
                        temp=torch.zeros((len(self.encoding_info[key])))
                        enc_idx=self.encoding_info[key].index(news_info['sourceDataInfo'][key])
                        temp[enc_idx]=1
                        
                        if idx == 0:
                            cat_tensor = temp
                        else:
                            cat_tensor = torch.cat([cat_tensor,temp],dim=0)
                            
                    return doc, label, cat_tensor, length_of_tokens
                
                else:
                    doc = self.transform(
                        title = news_info['labeledDataInfo']['newTitle'], 
                        text  = news_info['sourceDataInfo']['newsContent'].split('\n')
                        )
                
                    return doc, label

    def __len__(self):
        raise NotImplementedError

    def encoding_list(self):

        encoding = {}

        for key in self.cat_keys:
            encoding[key]=[]

        data_info = glob(os.path.join(self.data_dir,'*/*/*/*'))

        data = {}

        for filename in data_info:
            f = json.load(open(filename,'r'))
            data[filename] = f

        for x in data_info:
            info = data[x]
            for key in self.cat_keys:
                value=info['sourceDataInfo'][key]
                if value not in encoding[key]:
                    encoding[key].append(value)

        with open('./encoding_info.json','w') as jf:
            json.dump(encoding, jf)

        return encoding