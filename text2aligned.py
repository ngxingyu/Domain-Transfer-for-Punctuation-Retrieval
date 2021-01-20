import pandas as pd
import torch
import numpy as np
import regex as re
import argparse, os, csv
from datasets import load_dataset
import transformers

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
id2tag = {id: tag for tag, id in tag2id.items()}
tokenizer = transformers.ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')

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
def chunk_examples_with_degree(n):
    '''Ensure batched=True if using dataset.map or ensure the examples are wrapped in lists.'''
    def chunk_examples(examples):
        output={}
        output['texts']=[]
        output['tags']=[]
        for sentence in examples['transcript']:
            text,tag=text2masks(n)(sentence)
            output['texts'].append(text)
            output['tags'].append([0]+tag if text[0]!='' else tag) # [0]+tag so that in all case, the first tag refers to [CLS]
        return output
    return chunk_examples
assert(chunk_examples_with_degree(0)({'transcript':['Hello!Bye…']})=={'texts': [['Hello', 'Bye']], 'tags': [[0, 1, 9]]})

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

def chunk_to_len(max_length,tokens,labels):
    subwords,token_end_idxs = subword_tokenize(tokens)
    teim=token_end_idxs%(max_length-2)
    split_token_end_idxs=np.array_split(token_end_idxs,(np.argwhere((teim[1:])<teim[:-1]).flatten()+1).tolist())
    split_subwords=np.array_split(subwords,np.arange(max_length-2,len(subwords),max_length-2)) #token_end_idxs[np.argwhere((teim[1:])<teim[:-1]).flatten()+1].tolist()
    split_labels=np.array_split(labels[1:],(np.argwhere((teim[1:])<teim[:-1]).flatten()+1).tolist())
    ids=torch.tensor([pad_ids_to_len(max_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(_)+['[SEP]'])) for _ in split_subwords], dtype=torch.long)
    masks=torch.tensor([position_to_mask(max_length,_) for _ in split_token_end_idxs], dtype=torch.long)
    padded_labels=torch.tensor([pad_ids_to_len(max_length,[0]+list(_)+[0]) for _ in split_labels], dtype=torch.long)
    return ids,masks,padded_labels

def chunk_to_len_batch(max_length,tokens,labels, filename):
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    f=open(filename,"ab")
    for i,_ in enumerate(zip(tokens,labels)):
        # a,b,c=chunk_to_len(max_length,*_)
        # batch_ids.append(a)
        # batch_masks.append(b)
        # batch_labels.append(c)
        np.savetxt(f,torch.hstack([*chunk_to_len(max_length,*_)]),fmt='%i')
        print('.', end='')
    f.close()

def process_dataset(dataset, split, filename, max_length=128, overlap=63, degree=0, threads=1):
    data=dataset[split].map(chunk_examples_with_degree(degree), batched=True, batch_size=128,remove_columns=dataset[split].column_names, num_proc=threads)
    chunk_to_len_batch(max_length,data['texts'],data['tags'], filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enter csv file location. Extracts the "talk_id" and "transcript" column from csv and preprocesses transcript to the format for sentence punctuation prediction')
    parser.add_argument("-i", "--input", dest="filename", required=True,
                        help="input file (Omit .train .dev .test .csv)", metavar="FILE")
    parser.add_argument("-m", "--max", dest="max_length", required=False,
                        help="max sequence length", default=128, type=int)
    parser.add_argument("-o", "--overlap", dest="overlap_length", required=False,
                        help="max sequence length", default=63, type=int)
    parser.add_argument("-s", "--split", dest="splits", required=False,
                        help="single split, train dev test if empty", default='', type=str)
    parser.add_argument("-d", "--degree", dest="degree", required=False, help="Degree of labels", default=0, type=int)
    parser.add_argument("-t", "--multithread", dest="threads", required=False, help="num of threads", default=1, type=int)

    args = parser.parse_args()
    if (args.splits==''):
        splits=['train', 'dev', 'test']
    else:
        splits=[args.splits]
    for split in splits:
        paths=os.path.splitext(args.filename)
        filename=paths[0]+'.'+split+paths[1]
        print(validate_file(filename+'.csv'))
        ted=load_dataset('csv',data_files={split:filename+'.csv'})
        process_dataset(ted,split,filename+'-batched.csv',args.max_length,args.overlap_length,args.threads)
