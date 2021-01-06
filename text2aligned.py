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
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

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

def encode_tags(encodings, docs, max_length, overlap):
    encoded_labels = []
    doc_id=0
    label_offset=1
#     print(encodings.keys())
    for doc_offset,current_doc_id in zip(encodings.offset_mapping,encodings['overflow_to_sample_mapping']):
#         print(doc_id, end=' ')
        if current_doc_id>doc_id:
            doc_id+=1
            label_offset=0
            print('.', end='')
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * 0 #-100
        doc_enc_labels[0]=docs[doc_id][label_offset-1] # Set leading punctuation class
#         print([id2tag[t] if t>0 else '' for t in docs[doc_id][label_offset:label_offset+len(doc_offset)]])
        arr_offset = np.array(doc_offset)
        arr_mask = (arr_offset[:,0] == 0) & (arr_offset[:,1] != 0) # Gives the labels that should be assigned punctuation
        doc_enc_labels[arr_mask] = docs[doc_id][label_offset:label_offset+sum(arr_mask)]
        encoded_labels.append(doc_enc_labels)
        label_offset+=sum(arr_mask[:max_length-overlap-1])
    return encoded_labels

def process_dataset(dataset, split, max_length=128, overlap=63):
    data=dataset[split].map(chunk_examples_with_degree(0), batched=True, batch_size=max_length,remove_columns=dataset[split].column_names)
    encodings=tokenizer(data['texts'], is_split_into_words=True, return_offsets_mapping=True,
              return_overflowing_tokens=True, padding=True, truncation=True, max_length=max_length, stride=overlap)
    labels=encode_tags(encodings, data['tags'], max_length, overlap)
    encodings.pop("offset_mapping")
    encodings.pop("overflow_to_sample_mapping")
    return PunctuationDataset(torch.tensor(encodings['input_ids'],dtype=torch.long),
        torch.tensor(encodings['attention_mask'],dtype=torch.long),
        torch.tensor(labels,dtype=torch.long))

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
        dataset=process_dataset(ted,split,args.max_length,args.overlap_length)
        torch.save(dataset, filename+'.pt')

