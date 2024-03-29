from torch.utils.data import IterableDataset, Dataset, get_worker_info
import gc
import numpy as np
from typing import List, Optional, Dict
from core.utils import chunk_examples_with_degree, chunk_to_len_batch
import pandas as pd
import os
import torch
import subprocess
from time import time
from itertools import cycle, chain, islice, repeat
from math import ceil
from collections import Counter

class PunctuationDomainDataset(IterableDataset):

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
        alpha_sub=0.4, 
        alpha_del=0.4,
        alpha_ins=0.4,
        alpha_swp=0,
        alpha_spl=0.4,
        stride=0,
    ):
        if not (os.path.exists(csv_file)):
            raise FileNotFoundError(
                f'{csv_file} not found. The 2nd column of the file contains the transcripts.'
            )

        data_dir = os.path.dirname(csv_file)
        filename = os.path.basename(csv_file)

        if not filename.endswith('.csv'):
            raise ValueError("{text_file} should have extension .csv")
        
        self.csv_file =   csv_file
        self.max_seq_length =   max_seq_length
        self.manual_len=manual_len
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
        self.alpha_sub=alpha_sub
        self.alpha_del=alpha_del
        self.alpha_ins=alpha_ins
        self.alpha_swp=alpha_swp
        self.alpha_spl=alpha_spl
        self.stride=stride
        if not (os.path.exists(self.target_file)):
            os.system(f"sed '1d' {self.csv_file} > {self.target_file}")
        self.set_num_samples(self.target_file, num_samples, manual_len)
    def __iter__(self):
        self.dataset=iter(pd.read_csv(
                self.target_file,
                skiprows=(0 % self.len)*self.num_samples,
                header=None,
                dtype=str,
                chunksize=self.num_samples,
                ))
        return self
        

    def __next__(self):
        batch = next(self.dataset)[1]
        complete=batch
        if self.stride>0:
            for i in range(1,self.max_seq_length//self.stride):
                l=batch.str.split().map(len).values
                a=self.stride*i*np.ones_like(l)
                b=l
                complete=complete.append(pd.DataFrame({'t':batch,'a':a,'b':b}).apply(lambda row: ' '.join(row.t.split()[row.a:row.b]),axis=1))
            # pp(batch.shape,complete.shape)
        batch=complete
        chunked=chunk_examples_with_degree(self.degree, self.punct_label_ids, self.label_map, self.tokenizer,self.alpha_sub, self.alpha_del,self.alpha_ins,self.alpha_swp,self.alpha_spl)(batch)
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
        self.num_samples=min(self.num_samples,self.total_samples)
        self.len = max(1,int(self.total_samples / self.num_samples))

        

    def __len__(self):
        return pp(self.len)
    
    def shuffle(self, randomize=True, seed=42):
        int(subprocess.Popen(['wc', '-l', self.target_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
        os.system('bash data/shuffle.sh -i {} -o {} -a {} -s {} -m {} -t {}'.format(self.target_file, self.target_file, ['true','false'][randomize], seed, '100M',self.tmp_path))
        int(subprocess.Popen(['wc', '-l', self.target_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
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
        for _ in range(min(20,self.len)):
            print('.',end='')
            ni=next(it)
            ct+=torch.bincount(ni['labels'].view(-1),minlength=len(self.punct_label_ids))
        return ct/sum(ct)




class PunctuationDomainDatasets(IterableDataset):

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
                 low_resource_labelled_count:int = 0,
                 alpha_sub=0,
                 alpha_del=0,
                 alpha_ins=0,
                 alpha_swp=0,
                 alpha_spl=0,
                 stride=0,
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
        self.stride=stride
        for path in labelled:
            if manual_len>0:
                self.ds_lengths.append(min(manual_len,int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])))
            else:
                self.ds_lengths.append(int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]))
        for path in unlabelled:
            if split=='train' and low_resource_labelled_count>0:
                if manual_len>0:
                    self.ds_lengths.append(min(manual_len,int(subprocess.Popen(['wc', '-l', f'{path}.labelled.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])))
                    self.ds_lengths.append(min(manual_len,int(subprocess.Popen(['wc', '-l', f'{path}.unlabelled.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])))
                else:
                    self.ds_lengths.append(int(subprocess.Popen(['wc', '-l', f'{path}.labelled.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]))
                    self.ds_lengths.append(int(subprocess.Popen(['wc', '-l', f'{path}.unlabelled.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]))
            else:
                if manual_len>0:
                    self.ds_lengths.append(min(manual_len,int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])))
                else:
                    self.ds_lengths.append(int(subprocess.Popen(['wc', '-l', f'{path}.{split}.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]))
        self.max_length=max(self.ds_lengths) 
        self.per_worker=int(self.max_length/self.num_workers)
        self.len=max(1,ceil(self.per_worker/num_samples))
        self.class_weights=None

        self.alpha_sub=alpha_sub
        self.alpha_del=alpha_del
        self.alpha_ins=alpha_ins
        self.alpha_swp=alpha_swp
        self.alpha_spl=alpha_spl
        self.stride=stride

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
                    pad_start=pad_start,
                    alpha_sub=self.alpha_sub,
                    alpha_del=self.alpha_del,
                    alpha_ins=self.alpha_ins,
                    alpha_swp=self.alpha_swp,
                    alpha_spl=self.alpha_spl,
                    stride=self.stride,)
            self.datasets.append(dataset)
            self.iterables.append(cycle(dataset))
            
        for i,path in enumerate(unlabelled):
            target=os.path.join(tmp_path,os.path.split(path)[1])
            if split=='train' and low_resource_labelled_count>0:
                dataset=PunctuationDomainDataset(
                        csv_file=f'{path}.unlabelled.{split}.csv', tokenizer=tokenizer,
                        num_samples=num_samples,max_seq_length=max_seq_length,
                        punct_label_ids=punct_label_ids,
                        label_map=label_map,domain=len(labelled)+i,labelled=False,
                        randomize=randomize,
                        target_file=f'{target}.unlabelled.{split}.{data_id}.csv',
                        tmp_path=tmp_path,
                        attach_label_to_end=attach_label_to_end,
                        no_space_label=no_space_label,
                        manual_len=manual_len,
                        pad_start=pad_start,
                        alpha_sub=self.alpha_sub,
                        alpha_del=self.alpha_del,
                        alpha_ins=self.alpha_ins,
                        alpha_swp=self.alpha_swp,
                        alpha_spl=self.alpha_spl,
                        stride=self.stride,)
                self.datasets.append(dataset)
                self.iterables.append(cycle(dataset))
                dataset=PunctuationDomainDataset(
                        csv_file=f'{path}.labelled.{split}.csv', tokenizer=tokenizer,
                        num_samples=num_samples,max_seq_length=max_seq_length,
                        punct_label_ids=punct_label_ids,
                        label_map=label_map,domain=len(labelled)+i,labelled=True,
                        randomize=randomize,
                        target_file=f'{target}.labelled.{split}.{data_id}.csv',
                        tmp_path=tmp_path,
                        attach_label_to_end=attach_label_to_end,
                        no_space_label=no_space_label,
                        manual_len=manual_len,
                        pad_start=pad_start,
                        alpha_sub=self.alpha_sub,
                        alpha_del=self.alpha_del,
                        alpha_ins=self.alpha_ins,
                        alpha_swp=self.alpha_swp,
                        alpha_spl=self.alpha_spl,
                        stride=self.stride,)
                self.datasets.append(dataset)
                self.iterables.append(cycle(dataset))
            else:
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
                        pad_start=pad_start,
                        alpha_sub=self.alpha_sub,
                        alpha_del=self.alpha_del,
                        alpha_ins=self.alpha_ins,
                        alpha_swp=self.alpha_swp,
                        alpha_spl=self.alpha_spl,
                        stride=self.stride,
                        )
                self.datasets.append(dataset)
                self.iterables.append(cycle(dataset))

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
            b={k:torch.cat([torch.repeat_interleave(d[k],max(1,min_batch/d[k].shape[0]),dim=0)[:min_batch] for d in ds], dim=0) for k in ['input_ids','attention_mask','subtoken_mask','labels','domain']}
            rand=torch.randperm(b['labels'].shape[0])
            return {k:v[rand] for k,v in b.items()}
        else:
            return {k:torch.cat([d[k] for d in ds], dim=0) for k in ['input_ids','attention_mask','subtoken_mask','labels','domain']}

    def __len__(self):
        return self.len

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
        chunked=chunk_examples_with_degree(self.degree, self.punct_label_ids, self.label_map,)(queries)
        self.features = chunk_to_len_batch(max_seq_length, tokenizer,chunked['texts'],chunked['tags'],attach_label_to_end=attach_label_to_end,no_space_label=no_space_label,pad_start=pad_start)
        self.attach_label_to_end=attach_label_to_end
        self.num_samples=num_samples

    def __len__(self):
        return math.ceil(len(self.all_input_ids)/self.num_samples)

    def __getitem__(self, idx):
        return {k:v for k,v in self.features.items()}
