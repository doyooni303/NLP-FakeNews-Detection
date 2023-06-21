from tqdm.auto import tqdm
import torch 
import os 
import yaml
import argparse 
import pdb

from dataset import create_tokenizer, create_dataset, create_dataloader

# 수정 부분
def save(split, dataloader, savedir, name):
    
    if 'DualBERT' in name:
        main_dict, ctg_dict = {}, {}
        label_list = []
        for i, (doc,label) in enumerate(tqdm(dataloader, desc=split)):
            # main
            if len(main_dict) == 0:
                for k in doc['main'].keys():
                    main_dict[k] = []
                
            for k in doc['main'].keys():
                main_dict[k].append(doc['main'][k])
            
            # category
            if len(ctg_dict) == 0:
                for k in doc['ctg'].keys():
                    ctg_dict[k] = []
                
            for k in doc['ctg'].keys():
                ctg_dict[k].append(doc['ctg'][k])


            label_list.append(label)

        for k in main_dict.keys():
            main_dict[k] = torch.cat(main_dict[k])

        for k in ctg_dict.keys():
            ctg_dict[k] = torch.cat(ctg_dict[k])

        label_list = torch.cat(label_list)

        torch.save({'doc':{'main':main_dict,'ctg':ctg_dict}, 'label':label_list}, os.path.join(savedir,f'{split}.pt'))
        
    elif 'CAT_CONT_LEN' in name or 'Multimodal_net' in name:
        
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




    else:
        doc_dict = {}
        label_list = []
        for i, (doc, label) in enumerate(tqdm(dataloader, desc=split)):
            if len(doc_dict) == 0:
                for k in doc.keys():
                    doc_dict[k] = []
                
            for k in doc.keys():
                doc_dict[k].append(doc[k])
            label_list.append(label)

        for k in doc_dict.keys():
            doc_dict[k] = torch.cat(doc_dict[k])
        label_list = torch.cat(label_list)

        torch.save({'doc':doc_dict, 'label':label_list}, os.path.join(savedir,f'{split}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')
    parser.add_argument('--save_directory', type=str, default=None, help='save directory for dataloader')
    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)

    # save directory
    directory_name = cfg['RESULT']['dataname'] if args.save_directory is None else args.save_directory
    savedir = os.path.join(cfg['RESULT']['savedir'], directory_name)
    os.makedirs(savedir, exist_ok=True)

    #!
    # tokenizer 
    tokenizer, word_embed = create_tokenizer(
        name                          = cfg['TOKENIZER']['name'], 
        vocab_path                    = cfg['TOKENIZER'].get('vocab_path', None), 
        max_vocab_size                = cfg['TOKENIZER'].get('max_vocab_size', None),
        pretrained_model_name_or_path = cfg['TOKENIZER']['pretrained_model_name_or_path']
    )

    for split in cfg['MODE']['save_list']:
        # try:
        
        dataset_config = {
            'name': cfg['DATASET']['name'],
            'data_path': cfg['DATASET']['data_path'],
            'direct_path': cfg['DATASET'].get('direct_path', None),
            'split': split,
            'tokenizer': tokenizer, 
            'saved_data_path': cfg['DATASET']['saved_data_path'],
            **cfg['DATASET']['PARAMETERS']
        }

        if 'CAT_KEYS' in cfg['DATASET']:
            dataset_config['cat_keys'] = cfg['DATASET']['CAT_KEYS']

        dataset = create_dataset(**dataset_config)
        
        dataloader = create_dataloader(
            dataset     = dataset, 
            batch_size  = cfg['TRAIN']['batch_size'], 
            num_workers = cfg['TRAIN']['num_workers'],
            shuffle     = False
        )

        # save
        save(split, dataloader, savedir, cfg['DATASET']['name'])
        # except:
        #     print(f'{split} folder does not exist')
