from pytorch_lightning import LightningDataModule
from torch import dtype
from data import PunctuationDataset, PunctuationDatasets
from typing import List
import pandas as pd
import os
import torch

class PunctuationDataModule(LightningDataModule):
    def __init__(self, 
            labelled: List[str], 
            unlabelled: List[str], 
            train_batch_size: int,
            max_seq_length:int = 256,
            val_batch_size:int = 256, 
            num_workers:int = 1,
            pin_memory:bool = False,
            drop_last:bool = False):
        #unlabelled=[], batch_size = 256, max_seq_length = 256, num_workers=1):
        super().__init__()
        self.labelled=labelled
        self.unlabelled=unlabelled
        self.num_domains=len(labelled)+len(unlabelled)
        self.train_batch_size = max(1,train_batch_size//self.num_domains)
        ic(self.train_batch_size)
        self.val_batch_size = max(1,val_batch_size//self.num_domains)
        self.max_seq_length = max_seq_length
        self.num_workers=num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_dataset={}
        self.dev_dataset={}
        self.test_dataset={}

    def setup(self, stage=None):
        for unlabelled,l in enumerate([self.labelled,self.unlabelled]):
            for i,p in enumerate(l):
                domain=i+unlabelled*len(self.labelled) #unlabelled domain is increasing after labelled
                try:
                    with open("{}.train-stride.csv".format(p),'r') as f:
                        s=len(f.readline().split(' '))//3
                except IOError:
                    print('nofile')
                    s=0
                ic(s)
                if (s!=self.max_seq_length):
                    os.system("cp {} {}".format(p+'.train-batched.csv',p+'.train-stride.csv'))
                    if (self.max_seq_length!=256):
                        ic('generating strides:',self.max_seq_length)
                        n=np.loadtxt(open(p+".train-stride.csv", "rb"))
                        np.savetxt(p+".train-stride.csv", self.with_stride_split(n,self.max_seq_length),fmt='%d')

                if stage=='fit' or None:
                    self.train_dataset[domain] = PunctuationDataset(p+'.train-stride.csv', num_samples=self.train_batch_size, max_seq_length=self.max_seq_length, domain = domain, labelled=bool(1-unlabelled))
                    self.dev_dataset[domain] =  PunctuationDataset(p+'.dev-batched.csv', num_samples=self.val_batch_size, max_seq_length=self.max_seq_length, domain = domain, labelled=bool(1-unlabelled))
                    ic(self.train_dataset[domain].shuffle(sorted=True))
                    ic(self.train_dataset[domain].shuffle())

                if stage == 'test' or stage is None:
                    self.test_dataset[domain] =  PunctuationDataset(p+'.test-batched.csv', num_samples=self.val_batch_size, max_seq_length=self.max_seq_length, domain = domain, labelled=bool(1-unlabelled))

    def shuffle(self):
        for dataset in self.train_dataset.values():
            dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(PunctuationDatasets(*self.train_dataset.values()),batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(PunctuationDatasets(*self.dev_dataset.values()),batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(PunctuationDatasets(*self.test_dataset.values()),batch_size=None,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=self.drop_last)

    def with_stride_split(n,l):
        def with_stride(t,l):
            a=t[0,0]
            z=t[0,-1]
            t=t[:,1:-1].flatten()
            t=np.trim_zeros(t,'b')
            s=t.shape[0]
            nh=-(-s//(l-2))
            f=np.zeros((nh*(l-2),1))  
            f[:s,0]=t
            return np.hstack([np.ones((nh,1))*a,np.reshape(f,(-1,l-2)),np.ones((nh,1))*z])
        s=n.shape[1]
        a,b,c=n[:,:s//3],n[:,s//3:2*s//3],n[:,2*s//3:]
        a,b,c=with_stride(a,l), with_stride(b,l), with_stride(c,l)
        c1=np.zeros(a.shape)
        c1[:c.shape[0],:]=c
        return np.hstack([a,b,c1])

