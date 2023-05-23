from .build_dataset import FakeDataset
import torch
from typing import List,Union

class DualBERTDataset(FakeDataset):
    def __init__(self, tokenizer, max_word_len: int, max_category_len: int):
        super(DualBERTDataset, self).__init__(tokenizer=tokenizer)

        self.max_word_len = max_word_len
        self.max_category_len = max_category_len
        self.vocab = self.tokenizer.vocab

        # special token index
        self.pad_idx = self.vocab[self.vocab.padding_token]
        self.cls_idx = self.vocab[self.vocab.cls_token]

    def transform(self, title: str, text: list, category:str, subcategory:str) -> dict:
        # main part
        main_sent_list = [title] + text
        main_src = [self.tokenizer(d_i) for d_i in main_sent_list]

        main_src = self.length_processing(main_src,'main')

        main_input_ids, main_token_type_ids, main_attention_mask = self.tokenize(main_src)

        
        # additional cateogry part
        category_sent_list = [category,subcategory]
        category_src = [self.tokenizer(d_i) for d_i in category_sent_list]

        category_src = self.length_processing(category_src,'category')

        category_input_ids, category_token_type_ids, category_attention_mask = self.tokenize(category_src)

        main={'input_ids':main_input_ids,
              'attention_mask':main_attention_mask,
              'token_type_ids':main_token_type_ids}
        
        ctg={'input_ids':category_input_ids,
             'attention_mask':category_attention_mask,
             'token_type_ids':category_token_type_ids}
        
        doc ={'main':main,'ctg':ctg}

        return doc


    def tokenize(self, src: list) -> List[torch.Tensor]:
        src_subtokens = [[self.vocab.cls_token] + src[0] + [self.vocab.sep_token]] + [sum(src[1:],[]) + [self.vocab.sep_token]]
        input_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src_subtokens]
        
        token_type_ids = self.get_token_type_ids(input_ids)
        token_type_ids = sum(token_type_ids,[])
        
        input_ids = [x for sublist in input_ids for x in sublist]
        
        input_ids, token_type_ids, attention_mask = self.padding_bert(
            input_ids       = input_ids,
            token_type_ids  = token_type_ids
        )
        
        return input_ids, token_type_ids, attention_mask

    def length_processing(self, src: list, input_type: str) -> list:
        # 3 is the number of special tokens. ex) [CLS], [SEP], [SEP]
        max_word_len = self.max_word_len -3 if input_type == 'main' else self.max_category_len
        
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

    def __getitem__(self, i: int) -> Union[dict, int]:
        if self.saved_data_path:

            main = {}
            for k in self.data['doc']['main'].keys():
                main[k] = self.data['doc']['main'][k][i]
            
            ctg = {}
            for k in self.data['doc']['ctg'].keys():
                ctg[k] = self.data['doc']['ctg'][k][i]

            label = self.data['label'][i]

            return {'main':main,'ctg':ctg}, label
        
        else:
            news_info = self.data[self.data_info[i]]
        
            # label
            label = 1 if 'NonClickbait_Auto' not in self.data_info[i] else 0
        
            # transform and padding
            doc = self.transform(
                title = news_info['labeledDataInfo']['newTitle'], 
                text  = news_info['sourceDataInfo']['newsContent'].split('\n'),
                category = news_info['sourceDataInfo']['newsCategory'],
                subcategory = news_info['sourceDataInfo']['newsSubcategory'],
            )

            return doc, label

    def __len__(self):
        # self.data = {'doc':{'main':{'input_ids:...,},'ctg':{'input_ids':..},
        # 'label': tensor([1,1,...])}}


        if self.saved_data_path:
            return len(self.data['doc']['main']['input_ids'])
        else:
            return len(self.data)
    



