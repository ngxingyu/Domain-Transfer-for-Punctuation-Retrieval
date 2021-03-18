#!/usr/bin/env python
#%%
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import regex as re
import argparse, os, csv
from datasets import load_dataset
import transformers
import subprocess

class PunctuationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.input_ids[idx],dtype=torch.long),
        'attention_mask': torch.tensor(self.attention_mask[idx],dtype=torch.long),
        'labels': torch.tensor(self.labels[idx],dtype=torch.long)
        }
        return item
    def __len__(self):
        return len(self.labels)

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

tags=sorted(list('.?!,;:-—…'))
tag2id = {tag: id+1 for id, tag in enumerate(tags)}
tag2id['']=0
id2tag = {id: tag for tag, id in tag2id.items()}
tokenizer = transformers.ElectraTokenizerFast.from_pretrained('/home/nxingyu2/data/electra-base-discriminator')

def text2masks(n):
    def text2masks(text):
        '''Converts single paragraph of text into a list of words and corresponding punctuation based on the degree requested.'''
        if n==0: 
            refilter="(?<=[.?!,;:\-—… ])(?=[^.?!,;:\-—… ])|$"
        else:
            refilter="[.?!,;:\-—…]{1,%d}(?= *[^.?!,;:\-—…]+|$)|(?<=[^.?!,;:\-—…]) +(?=[^.?!,;:\-—…])"%(n)
        word=re.split(refilter,text, flags=re.V1)
        punct=re.findall(refilter,text, flags=re.V1)
        wordlist,punctlist=([] for _ in range(2))
        for i in zip(word,punct+['']):
            w,p=i[0].strip(),i[1].strip()
            if w!='':
                wordlist.append(re.sub(r'[.?!,;:\-—… ]','',w))
                punctlist.append(0 if not w[-1] in '.?!,;:-—…' else tag2id[w[-1]])
            if p!='':
                wordlist.append(p)
                punctlist.append(0)
        return(wordlist,punctlist)
    return text2masks
assert(text2masks(0)('"Hello!!')==(['"Hello'], [1]))
assert(text2masks(1)('"Hello!!')==(['"Hello', '!'], [1, 0]))
assert(text2masks(0)('"Hello!!, I am human.')==(['"Hello','I','am','human'], [2,0,0,4]))
assert(text2masks(2)('"Hello!!, I am human.')==(['"Hello', '!,','I','am','human','.'], [1,0,0,0,0,0]))

#%%
def chunk_examples_with_degree(n):
    '''Ensure batched=True if using dataset.map or ensure the examples are wrapped in lists.'''
    def chunk_examples(examples):
        output={}
        output['texts']=[]
        output['tags']=[]
        for sentence in examples:
            text,tag=text2masks(n)(sentence)
            output['texts'].append(text)
            output['tags'].append(tag)
            # output['tags'].append([0]+tag if text[0]!='' else tag) # [0]+tag so that in all case, the first tag refers to [CLS]
            # not necessary since all the leading punctuations are stripped
        return output
    return chunk_examples
assert(chunk_examples_with_degree(0)(['Hello!Bye…'])=={'texts': [['Hello', 'Bye']], 'tags': [[1, 9]]})

#%%
def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def subword_tokenize(tokens):
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = list(flatten(subwords))
    token_end_idxs = np.cumsum([0]+subword_lengths[:-1])+np.array(subword_lengths)-1
    return subwords, token_end_idxs

def position_to_mask(max_length,indices):
    o=np.zeros(max_length,dtype=np.int)
    o[indices%(max_length-2)+1]=1
    return o

def pad_ids_to_len(max_length,ids):
    o=np.zeros(max_length, dtype=np.int)
    o[:len(ids)]=np.array(ids)
    return o

def labels_to_position(mask,labels):
    mask[mask>0]=torch.tensor(labels)
    return mask.tolist()

def chunk_to_len(max_length,tokens,labels):
    subwords,token_end_idxs = subword_tokenize(tokens)
    teim=token_end_idxs%(max_length-2)
    split_token_end_idxs=np.array_split(token_end_idxs,(np.argwhere((teim[1:])<teim[:-1]).flatten()+1).tolist())
    split_subwords=np.array_split(subwords,np.arange(max_length-2,len(subwords),max_length-2)) #token_end_idxs[np.argwhere((teim[1:])<teim[:-1]).flatten()+1].tolist()
    split_labels=np.array_split(labels,(np.argwhere((teim[1:])<teim[:-1]).flatten()+1).tolist())
    ids=torch.tensor([pad_ids_to_len(max_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(_)+['[SEP]'])) for _ in split_subwords], dtype=torch.long)
    masks=[position_to_mask(max_length,_) for _ in split_token_end_idxs]
    padded_labels=torch.tensor([pad_ids_to_len(max_length,labels_to_position(*_)) for _ in zip(masks,split_labels)], dtype=torch.long)
    masks=torch.tensor(masks,dtype=torch.long)
    return ids,masks,padded_labels

def chunk_to_len_batch(max_length,tokens,labels, filename):
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    f=open(filename,"ab")
    for i,_ in enumerate(zip(tokens,labels)):
        a,b,c=chunk_to_len(max_length,*_)
        batch_ids.append(a)
        batch_masks.append(b)
        batch_labels.append(c)
        np.savetxt(f,torch.hstack([*chunk_to_len(max_length,*_)]),fmt='%i')
        print('.', end='')
    f.close()

def process_dataset(transcript, filename, max_length=128, overlap=63, degree=0, threads=1):
    data=chunk_examples_with_degree(degree)(transcript)
    chunk_to_len_batch(max_length,data['texts'],data['tags'], filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter csv file location. Extracts the "talk_id" and "transcript" column from csv and preprocesses transcript to the format for sentence punctuation prediction')
    parser.add_argument("-i", "--input", dest="path", required=True,
                        help="input file (Omit .train .dev .test .csv)", metavar="FILE")
    parser.add_argument("-m", "--max", dest="max_length", required=False,
                        help="max sequence length", default=128, type=int)
    parser.add_argument("-o", "--overlap", dest="overlap_length", required=False,
                        help="max sequence length", default=63, type=int)
    parser.add_argument("-s", "--split", dest="splits", required=False,
                        help="single split, train dev test if empty", default='', type=str)
    parser.add_argument("-d", "--degree", dest="degree", required=False, help="Degree of labels", default=0, type=int)
    # parser.add_argument("-t", "--multithread", dest="threads", required=False, help="num of threads", default=1, type=int)
    parser.add_argument('-c',"--chunksize", dest='chunksize', type=int, required=False, default=2000)

    args = parser.parse_args()
    if (args.splits==''):
        splits=['train', 'dev', 'test']
    else:
        splits=[args.splits]
    for split in splits:
        filename=args.path+'.'+split
        output_filename=filename+'-batched.csv'
        print(validate_file(filename+'.csv'))
        nb_samples=int(subprocess.Popen(['wc', '-l', filename+'.csv'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0])
        total = int((nb_samples+args.chunksize) / args.chunksize)
        print('maxlen',args.max_length)
        open(output_filename, 'w').close()
        o=pd.read_csv(filename+'.csv',
                  dtype='str',
                  header=None,
                  chunksize=args.chunksize)
        for i in tqdm(o,total=total):
            process_dataset(i[1],output_filename)

        # paths=os.path.splitext(args.filename)
        
        # ted=load_dataset('csv',data_files={split:filename+'.csv'}, column_names=['id','transcript'])
        
        # process_dataset(ted,split,filename+'-batched.csv',max_length=args.max_length,overlap=args.overlap_length,degree=args.degree, threads=args.threads)

# %%
