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
            test_unlabelled:bool = True
            ):
        #unlabelled=[], batch_size = 256, max_seq_length = 256, num_workers=1):
        super().__init__()
        self.labelled=labelled
        self.unlabelled=unlabelled
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer)
        self.punct_label_ids=punct_label_ids
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
        self.test_unlabelled=test_unlabelled
    
    def reset(self):
        self.train_dataset.__iter__()
        self.val_dataset.__iter__()
        self.test_dataset.__iter__()

    def setup(self, stage=None):
        if stage=='fit' or stage is None:
            self.train_dataset = PunctuationDomainDatasets(split='train',
                    num_samples=self.train_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    labelled=self.labelled,
                    unlabelled=self.unlabelled,
                    tokenizer=self.tokenizer,
                    randomize=self.train_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path)
            self.val_dataset = PunctuationDomainDatasets(split='dev',
                    num_samples=self.val_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    labelled=self.labelled,
                    unlabelled=self.unlabelled,
                    tokenizer=self.tokenizer,
                    randomize=self.val_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path)
        if stage=='test' or stage is None:
            if (len(self.unlabelled)>0) and self.test_unlabelled:
                self.test_dataset = PunctuationDomainDatasets(split='test',
                    num_samples=self.val_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    labelled=self.unlabelled,
                    unlabelled=[],
                    tokenizer=self.tokenizer,
                    randomize=self.val_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path
                    )
            else: self.test_dataset = PunctuationDomainDatasets(split='test',
                    num_samples=self.val_batch_size,
                    max_seq_length=self.max_seq_length,
                    punct_label_ids=self.punct_label_ids,
                    labelled=self.labelled,
                    unlabelled=self.unlabelled,
                    tokenizer=self.tokenizer,
                    randomize=self.val_shuffle,
                    data_id=self.data_id,
                    tmp_path=self.tmp_path
                    )

        logging.info(f"shuffling train set")
        # self.train_dataset.shuffle(randomize=False)
        self.train_dataset.shuffle(randomize=True, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)