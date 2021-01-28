# from torch.utils.data import Dataset
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType

import numpy as np
from typing import List, Optional, Dict
import pandas as pd
import os
import torch
import subprocess

class PunctuationDomainDataset(Dataset):

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
            "domain": NeuralType(('B'), ChannelType()),
        }

    def __init__(self, 
        csv_file:str, 
        tokenizer,
        num_samples:int=256,
        max_seq_length:int=256,
        punct_label_ids: Dict[str, int] = None,
        domain=0,
        labelled=True,
    ):
        if not (os.path.exists(csv_file)):
            raise FileNotFoundError(
                f'{csv_file} not found. The data should be joined in 1 csv file.\
                    Each line of the file contains the subword token ids, masks and class labels per row.'
            )

        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)

        if not filename.endswith('.csv'):
            raise ValueError("{text_file} should have extension .csv")
        # filename = filename[:-4]
        
        self.filepath = csv_file
        self.max_seq_length = max_seq_length
        self.set_num_samples(filepath, num_samples)
        self.domain=domain
        self.labelled=labelled
        self.tokenizer=tokenizer

    def __getitem__(self, idx):
        x = next(
            pd.read_csv(
                self.filepath,
                skiprows=(idx % self.len)*self.num_samples,
                chunksize=self.num_samples,
                header=None,
                delimiter=' '))
        x = torch.from_numpy(x.values).reshape(-1,3,self.max_seq_length) #x.shape[-1]//3
        return {'input_ids': torch.as_tensor(x[:,0,:], dtype=torch.long),
                'attention_mask': torch.as_tensor(x[:,1,:],dtype=torch.bool)if self.labelled else torch.zeros_like(x[:,1,:],dtype=torch.bool),
                'labels': torch.as_tensor(x[:,2,:],dtype=torch.long),
                'domain':self.domain*torch.ones(x.shape[0],1,dtype=torch.long)}

    def set_num_samples(self,filepath,num_samples):
        self.num_samples = num_samples
        self.total_samples=int(subprocess.Popen(['wc', '-l', filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
        self.len = int(self.total_samples / self.num_samples)
        

    def __len__(self):
        return self.len
    
    def view(d)->list:
        """:param d(dictionary): returns readable format of single input_ids and labels in the form of readable text"""
        a,_,c=d.values()
        return [' '.join([_[0]+_[1] for _ in list(zip(self.tokenizer.convert_ids_to_tokens(_[0]),[id2tag[id] for id in _[1].tolist()]))]) for _ in zip(a,c)]
    
    def shuffle(self, sorted=False, seed=42):
        os.system('bash data/shuffle.sh -i {} -o {} -a {} -s {}'.format(self.filepath, self.filepath, ['false','true'][sorted], seed))

class PunctuationDomainDatasets(Dataset):
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
            "domain": NeuralType(('B'), ChannelType()),
        }

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        b={k:torch.vstack([d[i][k] for d in self.datasets]) for k in ['input_ids','attention_mask','labels','domain']}
        rand=torch.randperm(b['labels'].size()[0])
        return {k:v[rand] for k,v in b.items()}

    def __len__(self):
        return max(len(d) for d in self.datasets)

class BertPunctuationInferDataset(Dataset):
    """
    Creates dataset to use during inference for punctuation and capitalization tasks with a pretrained model.
    For dataset to use during training with labels, see BertPunctuationCapitalizationDataset.
    Args:
        queries file to sequences, each line should a sentence, no header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'attention_mask': NeuralType(('B', 'T'), MaskType()),
        }

    def __init__(self, queries: List[str], max_seq_length: int, tokenizer: TokenizerSpec):
        """ Initializes BertPunctuationInferDataset. """
        features = get_features(queries=queries, max_seq_length=max_seq_length, tokenizer=tokenizer)
        self.all_input_ids = features['input_ids']
        self.all_attention_mask = features['attention_mask']

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return {'input_ids':self.all_input_ids[idx],
            'attention_mask':self.all_attention_mask[idx],
        )

def get_features(
    queries:str, 
    max_seq_length:int,
    tokenizer,
    punct_label_ids: dict = None,):

    def flatten(list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def pad_ids_to_len(max_length,ids):
        o=np.zeros(max_length, dtype=np.int)
        o[:len(ids)]=np.array(ids)
        return o

    def position_to_mask(max_length,indices):
        o=np.zeros(max_length,dtype=np.int)
        o[indices%(max_length-2)+1]=1
        return o

    batch_ids=[]
    batch_masks=[]
    for query in queries:
        wordlist=re.split('[^a-zA-Z0-9]+',query)
        subwords=list(map(tokenizer.tokenize,wordlist))
        subword_lengths=list(map(len,subwords))
        subwords=list(flatten(subwords))
        token_end_idxs=np.cumsum([0]+subword_lengths[:-1])+np.array(subword_lengths)-1
        teim=token_end_idxs%(max_seq_length-2)
        split_token_end_idxs=np.array_split(token_end_idxs,(np.argwhere((teim[1:])<teim[:-1]).flatten()+1).tolist())
        split_subwords=np.array_split(subwords,np.arange(max_length-2,len(subwords),max_seq_length-2)) 
        ids=torch.tensor([pad_ids_to_len(max_seq_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(_)+['[SEP]'])) for _ in split_subwords], dtype=torch.long)
        masks=[position_to_mask(max_length,_) for _ in split_token_end_idxs]
        batch_ids.append(ids)
        batch_masks.append(masks)
    return {'input_ids': torch.as_tensor(batch_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(batch_masks,dtype=torch.bool)}

    


        
            