
from tqdm.auto import tqdm
import torch 
import os 
import yaml
import argparse 
import pdb
from dataset import create_tokenizer, create_dataset, create_dataloader

def save(split, dataloader, savedir):
    
    doc_dict = {}
    
    for i, (doc, label, cat, length) in enumerate(tqdm(dataloader, desc=split)):
        if len(doc_dict) == 0:
            for k in doc.keys():
                doc_dict[k] = []
            
        for k in doc.keys():
            doc_dict[k].append(doc[k])

        if i==0:
            cat_doc = cat
            # cont_doc = cont
            label_list = label
            list_of_length = length
        else:
            cat_doc=torch.cat([cat_doc,cat],dim=0)
            # cont_doc=torch.cat([cont_doc,cont],dim=0)
            label_list=torch.cat([label_list,label],dim=0)
            list_of_length=torch.cat([list_of_length,length],dim=0)
            # ps -eo user,pid,ppid,rss,size,vsize,pmem,pcpu,time,comm --sort -rss | head -n 10
        
    for k in doc_dict.keys():
        doc_dict[k] = torch.cat(doc_dict[k])

    torch.save({'doc':doc_dict, 'label':label_list,
                'cat_doc': cat_doc, 'length_of_tokens': list_of_length},
                os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')
    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['RESULT']['dataname'])
    os.makedirs(savedir, exist_ok=True)

    # tokenizer
    tokenizer, word_embed = create_tokenizer(
        name            = cfg['TOKENIZER']['name'], 
        vocab_path      = cfg['TOKENIZER'].get('vocab_path', None), 
        max_vocab_size  = cfg['TOKENIZER'].get('max_vocab_size', None)
    )

    for split in ['train', 'validation', 'test']:
        dataset = create_dataset(
            name           = cfg['DATASET']['name'], 
            data_path      = cfg['DATASET']['data_path'], 
            direct_path    = cfg['DATASET'].get('direct_path',None),
            split          = split, 
            tokenizer      = tokenizer, 
            saved_data_path = cfg['DATASET']['saved_data_path'],
            cat_keys = cfg['DATASET']['CAT_KEYS'],
            **cfg['DATASET']['PARAMETERS']
        )
        # pdb.set_trace()
        dataloader = create_dataloader(
            dataset     = dataset, 
            batch_size  = cfg['TRAIN']['batch_size'], 
            num_workers = cfg['TRAIN']['num_workers'],
            shuffle     = False
        )
        # pdb.set_trace()
        # save
        save(split, dataloader, savedir)