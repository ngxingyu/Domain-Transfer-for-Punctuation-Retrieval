import numpy as np
import torch
from torch import nn
import regex as re
import snoop
from copy import deepcopy

__all__ = ['chunk_examples_with_degree', 'chunk_to_len_batch', 'view_aligned']

def flatten(list_of_lists):
    for l in list_of_lists:
        for item in l:
            yield item

def pad_to_len(max_seq_length,ids):
    '''[0, 1, 2] -> array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])'''
    o=np.zeros(max_seq_length, dtype=np.int)
    o[:len(ids)]=np.array(ids)
    return o

def position_to_mask(max_seq_length:int,indices:list):
    '''[0, 2, 5] -> array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0])'''
    assert(isinstance(max_seq_length,int))

    o=np.zeros(max_seq_length,dtype=np.int)
    try:
        o[np.array(indices)%(max_seq_length-2)+1]=1
    except:
        pp('position_to_mask',np.array(indices)%(max_seq_length-2)+1)
        o[(np.array(indices)%(max_seq_length-2)+1).astype(int)]=1
    return o

def align_labels_to_mask(mask,labels):
    '''[0,1,0],[2] -> [0,2,0]'''
    assert(sum(mask)==len(labels))
    m1=mask.copy()
    m1[mask>0]=torch.tensor(labels)
    return m1.tolist()

def view_aligned(texts,tags,tokenizer,labels_to_ids):
        return [re.sub(r'( ?\[((PAD)|(CLS)|(SEP))\] ?)',' ',re.sub(' +##','',' '.join( #[.?!,;:\-—… ]+
            [_[0]+_[1] for _ in list(
                zip(tokenizer.convert_ids_to_tokens(_[0]),
                    [labels_to_ids[id] for id in _[1].tolist()])
            )]
        ))) for _ in zip(texts,tags)]

def text2masks(n, labels_to_ids):
    def text2masks(text):
        '''Converts single paragraph of text into a list of words and corresponding punctuation based on the degree requested.'''
        if n==0: 
            refilter="(?<=[.?!,;:\-—… ])(?=[^.?!,;:\-—… ])|$"
        else:
            refilter="[.?!,;:\-—…]{1,%d}(?= *[^.?!,;:\-—…]+|$)|(?<=[^.?!,;:\-—…]) +(?=[^.?!,;:\-—…])"%(n)
        text=re.sub(r'^[_\W]*','',text)
        word=re.split(refilter,text, flags=re.V1)
        punct=re.findall(refilter,text, flags=re.V1)
        wordlist,punctlist=([] for _ in range(2))
        if word[-1]=='': # ensures text aligns
            word.pop()
        else:
            punct.append('')
        
        for i in zip(word,punct): #+[''] to correspond to the last word or '' after the last punctuation.
            w,p=i[0].strip(),i[1].strip()
            if w!='':
                wordlist.append(re.sub(r'[.?!,;:\-—… ]','',w))
                punctlist.append(0 if not w[-1] in '.?!,;:-—…' else labels_to_ids[w[-1]])
            if p!='':
                wordlist.append(p)
                punctlist.append(0)
        return(wordlist,punctlist)
    return text2masks

def chunk_examples_with_degree(n, labels_to_ids):
    '''Ensure batched=True if using dataset.map or ensure the examples are wrapped in lists.'''
    def chunk_examples(examples):
        output={}
        output['texts']=[]
        output['tags']=[]
        for sentence in examples:
            text,tag=text2masks(n, labels_to_ids)(sentence)
            output['texts'].append(text)
            output['tags'].append(tag)
            # output['tags'].append([0]+tag if text[0]!='' else tag) # [0]+tag so that in all case, the first tag refers to [CLS]
            # not necessary since all the leading punctuations are stripped
        return output
    return chunk_examples
assert(chunk_examples_with_degree(0,{'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, ';': 6, '?': 7, '—': 8, '…': 9})(['Hello!Bye…'])=={'texts': [['Hello', 'Bye']], 'tags': [[1, 9]]})

def subword_tokenize(tokenizer,tokens):
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = list(flatten(subwords))
    token_start_idxs = np.cumsum([0]+subword_lengths[:-1])
    token_end_idxs = np.cumsum([0]+subword_lengths[:-1])+np.array(subword_lengths)-1
    return subwords, token_start_idxs,token_end_idxs

def chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,tokens,labels=None):
    subwords,token_start_idxs,token_end_idxs = subword_tokenize(tokenizer,tokens)
    teim=token_end_idxs%(max_seq_length-2) if attach_label_to_end else token_start_idxs%(max_seq_length-2)

    breakpoints=(np.argwhere(teim[1:]<teim[:-1]).flatten()+1).tolist()
    split_token_idxs=np.array_split(token_end_idxs,breakpoints) if attach_label_to_end else np.array_split(token_start_idxs,breakpoints)
    split_subwords=np.array_split(subwords,np.arange(max_seq_length-2,len(subwords),max_seq_length-2))
    ids=[pad_to_len(max_seq_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(_)+['[SEP]'])) for _ in split_subwords]
    masks=[]
    for _ in split_token_idxs:
        masks.append(position_to_mask(max_seq_length,_).copy())
    padded_labels=None
    if labels!=None:
        split_labels=np.array_split(labels,breakpoints)
        padded_labels=[pad_to_len(max_seq_length,align_labels_to_mask(*_)) for _ in zip(masks,split_labels)]
    return ids,masks,padded_labels
    
def chunk_to_len_batch(max_seq_length,tokenizer,tokens,labels=None,labelled=True,ignore_index=-100, attach_label_to_end=None):
    no_mask=False
    if attach_label_to_end is None:
        no_mask=True
        attach_label_to_end=True
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    for i,_ in enumerate(zip(tokens,tokens) if labels==None else zip(tokens,labels)):
        a,b,c=chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,*_) if labels else chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,_[0])
        batch_ids.extend(a)
        batch_masks.extend(b)
        if labelled==True:
            batch_labels.extend(c)
    output = {'input_ids': torch.as_tensor(batch_ids, dtype=torch.long),
              'attention_mask': torch.as_tensor(batch_ids, dtype=torch.bool),
              'subtoken_mask': torch.as_tensor(batch_masks,dtype=torch.bool)}
    if no_mask:
        output['subtoken_mask']=output['attention_mask']&(output['input_ids']!=102)
    else:
        output['subtoken_mask']|=(output['input_ids']==101)  # dont want end token |(output['input_ids']==102)
        output['subtoken_mask']&=labelled
    output['labels']=torch.as_tensor(batch_labels,dtype=torch.long) if labelled==True else torch.zeros_like(output['input_ids'],dtype=torch.long)
    return output


def transformer_weights_init(module, std_init_range=0.02, xavier=True):
    """
    Initialize different weights in Transformer model.
    Args:
        module: torch.nn.Module to be initialized
        std_init_range: standard deviation of normal initializer
        xavier: if True, xavier initializer will be used in Linear layers
            as was proposed in AIAYN paper, otherwise normal initializer
            will be used (like in BERT paper)
    """

    if isinstance(module, nn.Linear):
        if xavier:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)