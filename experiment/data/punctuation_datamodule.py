from pytorch_lightning import LightningDataModule
from data import PunctuationDomainDataset, PunctuationDomainDatasets
from transformers import AutoTokenizer
from typing import List, Dict
import logging
from torch.utils.data import DataLoader
import torch
import os

class PunctuationDataModule(LightningDataModule):
    def __init__(self, 
            tokenizer:str,
            labelled: List[str], 
            unlabelled: List[str], 
            punct_label_ids: Dict[str,int],
            label_map:Dict[str,str],
            train_batch_size: int = 16,
            max_seq_length:int = 256,
            val_batch_size:int = 256, 
            num_workers:int = 1,
            pin_memory:bool = False,
            drop_last:bool = False, #Not Implemented
            train_shuffle:bool = True,
            val_shuffle:bool = False,
            seed: int = 42,
            data_id: str = '',
            tmp_path:str = '~/data/tmp',
            val_unlabelled:bool = True,
            test_unlabelled:bool = True,
            attach_label_to_end:bool = None,
            manual_len:int = 0,
            no_space_label: str = None,
            pad_start:int = 0,
            low_resource_labelled_count: int = 0,
            ):
        #unlabelled=[], batch_size = 256, max_seq_length = 256, num_workers=1):
        super().__init__()
        self.labelled=labelled
        self.unlabelled=unlabelled
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer)
        self.punct_label_ids=pp(punct_label_ids)
        self.label_map=label_map
        self.num_domains=len(labelled)+len(unlabelled)
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        # self.train_batch_size = max(1,train_batch_size//self.num_domains)
        # logging.info(f"using training batch_size of {self.train_batch_size} for each domain")
        # self.val_batch_size = max(1,val_batch_size//self.num_domains)
        # logging.info(f"using dev batch_size of {self.train_batch_size} for each domain")
        self.max_seq_length = max_seq_length
        self.num_workers=num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_shuffle=train_shuffle
        self.val_shuffle=val_shuffle
        self.train_dataset=None
        self.dev_dataset=None
        self.test_dataset=None
        self.seed=seed
        self.data_id=data_id
        self.tmp_path=tmp_path
        self.val_unlabelled=val_unlabelled
        self.test_unlabelled=test_unlabelled
        self.attach_label_to_end=attach_label_to_end
        self.manual_len=manual_len
        self.pad_start=pad_start
        try:
            self.no_space_label=self.punct_label_ids[no_space_label]
        except:
            self.no_space_label=None
        self.low_resource_labelled_count = low_resource_labelled_count
    
    def reset(self):
        # self.setup('fit')
        self.train_dataset=iter(self.train_dataset)
        self.val_dataset=iter(self.val_dataset)
        self.test_dataset=iter(self.test_dataset)

    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            self.train_dataset = PunctuationDomainDatasets(split='train',
                    num_samples=self.train_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    label_map=self.label_map,
                    labelled=self.labelled,
                    unlabelled=self.unlabelled,
                    tokenizer=self.tokenizer,
                    randomize=self.train_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path,
                    attach_label_to_end=self.attach_label_to_end,
                    manual_len=self.manual_len,
                    no_space_label=self.no_space_label,
                    pad_start=self.pad_start,
                    low_resource_labelled_count=self.low_resource_labelled_count,
                    )
            if (len(self.unlabelled)>0) and self.val_unlabelled:
                self.val_dataset = PunctuationDomainDatasets(split='dev',
                        num_samples=self.val_batch_size,
                        max_seq_length=self.max_seq_length,
                        punct_label_ids=self.punct_label_ids,
                        label_map=self.label_map,
                        labelled=self.unlabelled,
                        unlabelled=self.labelled,
                        tokenizer=self.tokenizer,
                        randomize=self.val_shuffle,
                        data_id=self.data_id,
                        tmp_path=self.tmp_path,
                        attach_label_to_end=self.attach_label_to_end,
                        no_space_label=self.no_space_label,
                        pad_start=self.pad_start,
                        )
            else:
                self.val_dataset = PunctuationDomainDatasets(split='dev',
                        num_samples=self.val_batch_size,
                        max_seq_length=self.max_seq_length,
                        punct_label_ids=self.punct_label_ids,
                        label_map=self.label_map,
                        labelled=self.labelled,
                        unlabelled=self.unlabelled,
                        tokenizer=self.tokenizer,
                        randomize=self.val_shuffle,
                        data_id=self.data_id,
                        tmp_path=self.tmp_path,
                        attach_label_to_end=self.attach_label_to_end,
                        no_space_label=self.no_space_label,
                        pad_start=self.pad_start,
                        )
        if stage=='test' or stage is None:
            if (len(self.unlabelled)>0) and self.test_unlabelled:
                self.test_dataset = PunctuationDomainDatasets(split='test',
                    num_samples=self.val_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    label_map=self.label_map,
                    labelled=self.unlabelled,
                    unlabelled=[], #self.labelled
                    tokenizer=self.tokenizer,
                    randomize=self.val_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path,
                    attach_label_to_end=self.attach_label_to_end,
                    no_space_label=self.no_space_label,
                    pad_start=self.pad_start,
                    )
            else: self.test_dataset = PunctuationDomainDatasets(split='test',
                    num_samples=self.val_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    label_map=self.label_map,
                    labelled=self.labelled,
                    unlabelled=[], #self.unlabelled
                    tokenizer=self.tokenizer,
                    randomize=self.val_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path,
                    attach_label_to_end=self.attach_label_to_end,
                    no_space_label=self.no_space_label,
                    pad_start=self.pad_start,
                    )

        logging.info(f"shuffling train set")
        # self.train_dataset.shuffle(randomize=False)
        if (self.train_shuffle):
            self.train_dataset.shuffle(randomize=True, seed=self.seed)
        pp('finished setup')
        
    def train_dataloader(self):
        pp('train',self.num_workers)
        return DataLoader(self.train_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def val_dataloader(self):
        pp('val',self.num_workers)
        return DataLoader(self.val_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def test_dataloader(self):
        pp('test',self.num_workers)
        return DataLoader(self.test_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)