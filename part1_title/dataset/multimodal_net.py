from .build_dataset import FakeDataset
import torch 
from typing import List
import pdb
import numpy as np

class Multimodal_netDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int, name: str,
                 use_cat: bool = False):
        super(Multimodal_netDataset, self).__init__(tokenizer=tokenizer, name = name, use_cat = use_cat)

        self.max_word_len = max_word_len
        self.vocab = self.tokenizer.vocab
        
        # special token index
        self.pad_idx = self.vocab[self.vocab.padding_token]
        self.cls_idx = self.vocab[self.vocab.cls_token]
        
    def list_padding(self, lst, padding):
        return lst+[0 for i in range(padding-len(lst))]
    
    def transform(self, title: str, text: list) -> dict:
        
        # 원본 제목 + 본문 데이터
        sent_list = [title] + text
        # 각 요소를 토크나이즈
        src = [self.tokenizer(d_i) for d_i in sent_list]
        # max_word_len 기준으로 자르기
        src = self.length_processing(src)
        # 각 문장 별 토큰화 이후 length
        length_of_tokens = [len(i) for i in src]
        length_of_tokens = self.list_padding(length_of_tokens, 100)
        length_of_tokens = np.array(length_of_tokens)
        
        input_ids, token_type_ids, attention_mask = self.tokenize(src)

        doc = {}
        doc['input_ids'] = input_ids
        doc['attention_mask'] = attention_mask
        doc['token_type_ids'] = token_type_ids

        return length_of_tokens, doc


    def tokenize(self, src: list) -> List[torch.Tensor]:
        title_token_len = len(src[0])
        contents_token_len = []
        src_subtokens = [[self.vocab.cls_token] + src[0] + [self.vocab.sep_token]] + [sum(src[1:],[]) + [self.vocab.sep_token]]
        input_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src_subtokens]
        
        token_type_ids = self.get_token_type_ids(input_ids)
        token_type_ids = sum(token_type_ids,[])
        
        input_ids = [x for sublist in input_ids for x in sublist]
        # indice_of_last = [len(sublist) for sublist in input_ids]
        
        input_ids, token_type_ids, attention_mask = self.padding_bert(
            input_ids       = input_ids,
            token_type_ids  = token_type_ids
        )
        
        return input_ids, token_type_ids, attention_mask

    def length_processing(self, src: list) -> list:
        max_word_len = self.max_word_len - 3 # 3 is the number of special tokens. ex) [CLS], [SEP], [SEP]
        
        cnt = 0
        processed_src = []
        for sent in src:
            cnt += len(sent)
            if cnt > max_word_len:
                sent = sent[:len(sent) - (cnt-max_word_len)]
                processed_src.append(sent)
                break

            else:
                processed_src.append(sent)

        return processed_src


    def pad(self, data: list, pad_idx: int) -> list:
        data = data + [pad_idx] * max(0, (self.max_word_len - len(data)))
        return data


    def padding_bert(self, input_ids: list, token_type_ids: list) -> List[torch.Tensor]:
        # padding using bert models (bts, kobertseg)        
        input_ids = torch.tensor(self.pad(input_ids, self.pad_idx))
        token_type_ids = torch.tensor(self.pad(token_type_ids, self.pad_idx))

        attention_mask = ~(input_ids == self.pad_idx)

        return input_ids, token_type_ids, attention_mask


    def get_token_type_ids(self, input_ids: list) -> list:
        # for segment token
        token_type_ids = []
        for i, v in enumerate(input_ids):
            if i % 2 == 0:
                token_type_ids.append([0] * len(v))
            else:
                token_type_ids.append([1] * len(v))
        return token_type_ids


    def __len__(self):
        if self.saved_data_path:
            return len(self.data['doc']['input_ids'])
        else:
            return len(self.data)
    



