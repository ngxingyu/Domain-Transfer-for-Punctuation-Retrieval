from torch.utils.data import IterableDataset, Dataset, get_worker_info
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
import gc
import numpy as np
from typing import List, Optional, Dict
from core.utils import chunk_examples_with_degree, chunk_to_len_batch
import pandas as pd
import os
import torch
import subprocess
from time import time
from itertools import cycle, chain, islice

class PunctuationDomainDataset(IterableDataset):

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""

        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "subtoken_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
            "domain": NeuralType(('B'), ChannelType()),
        }

    def __init__(self, 
        csv_file:str, 
        tokenizer,
        num_samples:int=256,
        max_seq_length:int=256,
        degree=0,
        punct_label_ids: Dict[str, int] = None,
        label_map:Dict[str,str] = None,
        domain=0,
        labelled=True,
        randomize=True,
        target_file='',
        tmp_path='~/data/tmp',
        start=0,
        end=-1,
        attach_label_to_end=None,
        no_space_label=None,
        manual_len=0,
        pad_start=0,
    ):
        if not (os.path.exists(csv_file)):
            raise FileNotFoundError(
                f'{csv_file} not found. The 2nd column of the file contains the transcripts.'
            )

        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)

        if not filename.endswith('.csv'):
            raise ValueError("{text_file} should have extension .csv")
        # filename = filename[:-4]
        
        self.csv_file =   csv_file
        self.max_seq_length =   max_seq_length
        self.manual_len=manual_len
        self.set_num_samples(csv_file, num_samples, manual_len)
        self.domain=  domain
        self.punct_label_ids=punct_label_ids
        self.label_map=label_map
        self.labelled=  labelled
        self.tokenizer= tokenizer
        self.degree=degree
        self.randomize=randomize
        self.target_file=target_file
        self.tmp_path=tmp_path
        self.attach_label_to_end=attach_label_to_end
        self.no_space_label=no_space_label
        self.pad_start=pad_start
        if not (os.path.exists(self.target_file)):
            os.system(f"sed '1d' {self.csv_file} > {self.target_file}")

    def __iter__(self):
        self.dataset=iter(pd.read_csv(
                self.csv_file,
                skiprows=(0 % self.len)*self.num_samples,
                header=None,
                dtype=str,
                chunksize=self.num_samples,
                ))
        return self
        

    def __next__(self):
        batch = next(self.dataset)[1]

        l=batch.str.split().map(len).values
        n=8
        a=np.maximum((l-self.max_seq_length*n).clip(min=0),(l*np.random.random(l.__len__())).astype(int))
        b=np.minimum(l,a+self.max_seq_length*n)
        batch=pd.DataFrame({'t':batch,'a':a,'b':b}).apply(lambda row: ' '.join(row.t.split()[row.a:row.b]),axis=1)

        chunked=chunk_examples_with_degree(self.degree, self.punct_label_ids, self.label_map)(batch)
        batched=chunk_to_len_batch(self.max_seq_length,self.tokenizer,chunked['texts'],chunked['tags'],self.labelled,attach_label_to_end=self.attach_label_to_end,no_space_label=self.no_space_label, pad_start=self.pad_start)
        num_samples=batched['labels'].shape[0]
        batched['domain']=self.domain*torch.ones(num_samples,1,dtype=torch.long)
        gc.collect()
        if self.randomize:
            rand=torch.randperm(num_samples)
            return {k:v[rand] for k,v in batched.items()}
        else:
            return batched

    def set_num_samples(self,csv_file,num_samples, manual_len):
        self.num_samples = num_samples
        self.total_samples=int(subprocess.Popen(['wc', '-l', csv_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
        if manual_len>0:
            self.total_samples=min(manual_len,self.total_samples)
        self.len = int(self.total_samples / self.num_samples)
        

    def __len__(self):
        pp('dataset')
        return pp(self.len)
    
    def shuffle(self, randomize=True, seed=42):
        pp(os.system('bash data/shuffle.sh -i {} -o {} -a {} -s {} -m {} -t {}'.format(self.target_file, self.target_file, ['true','false'][randomize], seed, '100M',self.tmp_path)))
        self.dataset=iter(pd.read_csv(
                self.target_file,
                skiprows=(0 % self.len)*self.num_samples,
                header=None,
                dtype=str,
                chunksize=self.num_samples,
                ))
    
    def determine_class_weights(self):
        it=iter(self)
        ct=torch.zeros(len(self.punct_label_ids))
        for _ in range(20):
            print('.',end='')
            ni=next(it)
            ct+=torch.bincount(ni['labels'].view(-1),minlength=len(self.punct_label_ids))
        return ct/sum(ct)




class PunctuationDomainDatasets(IterableDataset):
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
            "subtoken_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
            "domain": NeuralType(('B'), ChannelType()),
        }

    def __init__(self, 
                 split:str,
                 num_samples:int,
                 max_seq_length:int,
                 punct_label_ids: Dict[str, int],
                 label_map:Dict[str,str],
                 labelled: List[str],
                 unlabelled: List[str],
                 tokenizer,
                 randomize:bool=True,
                 data_id='',
                 tmp_path='~/data/tmp',
                 attach_label_to_end=None,
                 manual_len:int=0,
                 no_space_label:int=None,
                 pad_start:int=0,
                 ):
        worker_info = get_worker_info()
        self.num_workers=1 if worker_info is None else worker_info.num_workers
        self.num_labelled=len(labelled)
        self.datasets = []
        self.iterables=[]
        self.randomize=randomize
        self.punct_label_ids=punct_label_ids
        self.label_map=label_map
        self.ds_lengths=[]
        self.labelled=labelled
        for path in labelled+unlabelled:
            if manual_len>0:
                self.ds_lengths.append(min(manual_len,int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])))
            else:
                self.ds_lengths.append(int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]))
        self.max_length=max(self.ds_lengths) 
        self.per_worker=int(self.max_length/self.num_workers)
        self.len=int(self.per_worker/num_samples) 
        self.class_weights=None


        for i,path in enumerate(labelled):
            target=os.path.join(tmp_path,os.path.split(path)[1])
            dataset=PunctuationDomainDataset(
                    csv_file=f'{path}.{split}.csv', tokenizer=tokenizer,
                    num_samples=num_samples,max_seq_length=max_seq_length,
                    punct_label_ids=punct_label_ids,
                    label_map=label_map,
                    domain=i,labelled=True,
                    randomize=randomize,
                    target_file=f'{target}.{split}.{data_id}.csv',
                    tmp_path=tmp_path,
                    attach_label_to_end=attach_label_to_end,
                    no_space_label=no_space_label,
                    manual_len=manual_len,
                    pad_start=pad_start,)
            self.datasets.append(dataset)
            self.iterables.append(cycle(dataset))
            
        for i,path in enumerate(unlabelled):
            target=os.path.join(tmp_path,os.path.split(path)[1])
            dataset=PunctuationDomainDataset(
                    csv_file=f'{path}.{split}.csv', tokenizer=tokenizer,
                    num_samples=num_samples,max_seq_length=max_seq_length,
                    punct_label_ids=punct_label_ids,
                    label_map=label_map,domain=len(labelled)+i,labelled=False,
                    randomize=randomize,
                    target_file=f'{target}.{split}.{data_id}.csv',
                    tmp_path=tmp_path,
                    attach_label_to_end=attach_label_to_end,
                    no_space_label=no_space_label,
                    manual_len=manual_len,
                    pad_start=pad_start,)
            self.datasets.append(dataset)
            self.iterables.append(cycle(dataset))

    # def __getitem__(self, i):
    #     ds=[d[i] for d in self.datasets]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        self.iterables=[]
        for ds_length, dataset in zip(self.ds_lengths,self.datasets):
            start = (worker_id*self.per_worker)%ds_length
            self.iterables.append(cycle(chain(islice(iter(dataset),start,None),islice(iter(dataset),start))))
        return self

    def __next__(self):
        ds=[next(d) for d in self.iterables]
        if self.randomize:
            min_batch=1000000
            for d in ds:
                size=d['domain'].shape[0]
                if size<min_batch:
                    min_batch=size
            #Ensure all domains are evenly represented
            b={k:torch.cat([d[k][:min_batch] for d in ds], dim=0) for k in ['input_ids','attention_mask','subtoken_mask','labels','domain']}
            rand=torch.randperm(b['labels'].shape[0])
            return {k:v[rand] for k,v in b.items()}
        else:
            return {k:torch.cat([d[k] for d in ds], dim=0) for k in ['input_ids','attention_mask','subtoken_mask','labels','domain']}

    def __len__(self):
        return pp(self.len)

    def shuffle(self, randomize=True, seed=42):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        if worker_id==0:
            for _ in self.datasets:
                print(f"shuffling {_}")
                _.shuffle(randomize,seed)
    
    def determine_class_weights(self):
        if self.class_weights is None:
            ct=torch.zeros(len(self.punct_label_ids))
            for _ in range(self.num_labelled):
                ct+=self.datasets[_].determine_class_weights()
            self.class_weights=self.num_labelled/ct
        return self.class_weights


class PunctuationInferenceDataset(Dataset):
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
            'subtoken_mask': NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self, 
    tokenizer, 
    queries: List[str], 
    max_seq_length: int, 
    punct_label_ids:Dict[str,int], 
    label_map:Dict[str,str], 
    num_samples:int=256, 
    degree:int = 0, 
    attach_label_to_end:bool=None,
    no_space_label=None,
    pad_start:int=0,
    ):
        """ Initializes BertPunctuationInferDataset. """
        self.degree=degree
        self.punct_label_ids=punct_label_ids
        self.label_map = label_map
        chunked=chunk_examples_with_degree(self.degree, self.punct_label_ids, self.label_map)(queries)
        self.features = chunk_to_len_batch(max_seq_length, tokenizer,chunked['texts'],chunked['tags'],attach_label_to_end=attach_label_to_end,no_space_label=no_space_label,pad_start=pad_start)
        self.attach_label_to_end=attach_label_to_end
        # self.all_input_ids = features['input_ids']
        # self.all_attention_mask = features['attention_mask']
        # self.all_subtoken_mask = features['subtoken_mask']
        self.num_samples=num_samples

    def __len__(self):
        return math.ceil(len(self.all_input_ids)/self.num_samples)

    def __getitem__(self, idx):
        # lower=idx*self.num_samples
        # print(idx)
        # (int(idx)+1)*self.num_samples+1)
        # upper=min(len(self.all_input_ids),(int(idx)+1)*self.num_samples+1)
        return {k:v for k,v in self.features.items()}
