from torch.utils.data import Dataset
from torch import dtype
from typing import List, Optional, Dict
import pandas as pd
import os
import torch
import subprocess

class PunctuationDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, dtype]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": torch.long,
            "attention_mask": torch.bool,
            "labels": torch.long,
            "domain": torch.long,
        }

    def __init__(self, 
        filepath:str, 
        num_samples=256,
        max_seq_length=256,
        domain=0,
        labelled=True
    ):
        self.filepath = filepath
        self.max_seq_length = max_seq_length
        self.set_num_samples(filepath, num_samples)
        self.domain=domain
        self.labelled=labelled

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
#         self.total_samples=int(os.system(f'wc -l {filepath}'))
        self.total_samples=int(subprocess.Popen(['wc', '-l', filepath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
        self.len = int(self.total_samples / self.num_samples)
        

    def __len__(self):
        return self.len
    
    def view(d)->list:
        """:param d(dictionary): returns readable format of single input_ids and labels in the form of readable text"""
        a,_,c=d.values()
        return [' '.join([_[0]+_[1] for _ in list(zip(eltok.convert_ids_to_tokens(_[0]),[id2tag[id] for id in _[1].tolist()]))]) for _ in zip(a,c)]
    
    def shuffle(self, sorted=False, seed=42):
        os.system('bash data/shuffle.sh -i {} -o {} -a {} -s {}'.format(self.filepath, self.filepath, ['false','true'][sorted], seed))

class PunctuationDatasets(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        b={k:torch.vstack([d[i][k] for d in self.datasets]) for k in ['input_ids','attention_mask','labels','domain']}
        rand=torch.randperm(b['labels'].size()[0])
        return {k:v[rand] for k,v in b.items()}

    def __len__(self):
        return max(len(d) for d in self.datasets)

