import numpy as np
import torch
import regex as re

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
    o=np.zeros(max_seq_length,dtype=np.int)
    o[np.array(indices)%(max_seq_length-2)+1]=1
    return o

def align_labels_to_mask(mask,labels):
    '''[0,1,0],[2] -> [0,2,0]'''
    assert(sum(mask)==len(labels))
    mask[mask>0]=torch.tensor(labels)
    return mask.tolist()

def view_aligned(texts,tags,tokenizer,labels_to_ids):
        return [re.sub(' ##','',' '.join(
            [_[0]+_[1] for _ in list(
                zip(tokenizer.convert_ids_to_tokens(_[0]),
                    [labels_to_ids[id] for id in _[1].tolist()])
            )]
        )) for _ in zip(texts,tags)]

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
    token_end_idxs = np.cumsum([0]+subword_lengths[:-1])+np.array(subword_lengths)-1
    return subwords, token_end_idxs

def chunk_to_len(max_seq_length,tokenizer,tokens,labels=None):
    subwords,token_end_idxs = subword_tokenize(tokenizer,tokens)
    teim=token_end_idxs%(max_seq_length-2)
    breakpoints=(np.argwhere(teim[1:]<teim[:-1]).flatten()+1).tolist()
    split_token_end_idxs=np.array_split(token_end_idxs,breakpoints)
    split_subwords=np.array_split(subwords,np.arange(max_seq_length-2,len(subwords),max_seq_length-2))
    ids=[pad_to_len(max_seq_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(_)+['[SEP]'])) for _ in split_subwords]
    masks=[position_to_mask(max_seq_length,_) for _ in split_token_end_idxs]
    padded_labels=None
    if labels!=None:
        split_labels=np.array_split(labels,breakpoints)
        padded_labels=[pad_to_len(max_seq_length,align_labels_to_mask(*_)) for _ in zip(masks,split_labels)]
    return ids,masks,padded_labels
    
def chunk_to_len_batch(max_seq_length,tokenizer,tokens,labels=None,labelled=True):
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    for i,_ in enumerate(zip(tokens,tokens) if labels==None else zip(tokens,labels)):
        a,b,c=chunk_to_len(max_seq_length,tokenizer,*_) if labels else chunk_to_len(max_seq_length,tokenizer,_[0])
        batch_ids.extend(a)
        batch_masks.extend(b)
        if labelled==True:
            batch_labels.extend(c)
    output = {'input_ids': torch.as_tensor(batch_ids, dtype=torch.long),
              'attention_mask': torch.as_tensor(batch_ids, dtype=torch.bool),
              'subtoken_mask': torch.as_tensor(batch_masks,dtype=torch.bool)*labelled}
    output['labels']=torch.as_tensor(batch_labels,dtype=torch.short) if labelled==True else torch.zeros_like(output['input_ids'],dtype=torch.short)
    return output