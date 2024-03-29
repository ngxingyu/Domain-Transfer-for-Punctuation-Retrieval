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
        o[(np.array(indices)%(max_seq_length-2)+1).astype(int)]=1
    return o

def align_labels_to_mask(mask,labels):
    '''[0,1,0],[2] -> [0,2,0]'''
    assert(sum(mask)==len(labels))
    m1=mask.copy()
    m1[mask>0]=torch.tensor(labels)
    return m1.tolist()

def view_aligned(texts,tags,tokenizer,labels_to_ids):
    '''Convert tokens ids and labels into readable text'''
    output=[re.sub(r'( ?\[((CLS))\] ?)',' ',
        re.sub(" \' ","\'",
            re.sub('# +','',
                re.sub('#? +##','',
                    ' '.join( #[.?!,;:\-—… ]+
                        [_[0]+_[1] for _ in list(
                            zip(tokenizer.convert_ids_to_tokens(_[0]),
                                [labels_to_ids[id] for id in _[1].tolist()])
                            )
                        ]
                    )
                )
            )
        )
    ) for _ in zip(texts,tags)]
    newoutput=[]
    prevappend=False
    for value in output:
        if value[-5:]!='[PAD]':
            append=True
        else:
            append=False
        value=re.sub(r'( ?\[((SEP)|(PAD))\] ?)',' ',value).strip()
        if prevappend:
            newoutput[-1]=newoutput[-1]+'// '+value
        else:
            newoutput.append(value)
        prevappend=append
    return newoutput

def text2masks(n, labels_to_ids,label_map):
    '''closure for text2masks specifying degree (degree 1 refers to the last punct., 2 - the 2nd last etc.)'''
    def text2masks(text):
        '''Converts single paragraph of text into a list of words and corresponding punctuation based on the degree requested.'''
        labels=''.join(labels_to_ids.keys())
        if n==0: 
            refilter=f"(?<=[{labels} ])(?=[^{labels} ])|$"
        else:
            refilter=f"[{labels}]{{1,{n}}}(?= *[^{labels}]+|$)|(?<=[^{labels}]) +(?=[^{labels}])"
        # text=re.sub(f'…[\w {labels}]{0,15}…','…',text)
        text=re.sub(r'^[_\W]*','',text)
        if label_map is not None:
            for k,v in label_map.items():
                text=re.sub(f"(?<=[{labels} ]){k}+","",text)
                text=re.sub(f"(?<=[A-Za-z0-9 ]){k}+",v,text)
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
                wordlist.append(re.sub(f'[{labels} ]','',w))
                punctlist.append(0 if not w[-1] in labels else labels_to_ids[w[-1]])
            if p!='':
                wordlist.append(p)
                punctlist.append(0)
        return(wordlist,punctlist)
    return text2masks

def chunk_examples_with_degree(n, labels_to_ids,label_map,tokenizer=None,alpha_sub=0.4, alpha_del=0.4, alpha_ins=0.4,alpha_swp=0,alpha_spl=0):
    '''Ensure batched=True if using dataset.map or ensure the examples are wrapped in lists.'''
    def chunk_examples(examples):
        output={}
        output['texts']=[]
        output['tags']=[]
        for sentence in examples:
            if tokenizer is not None:
                sentence=all_transform(sentence,tokenizer,alpha_sub, alpha_del, alpha_ins,alpha_swp,alpha_spl)
            text,tag=text2masks(n, labels_to_ids, label_map)(sentence)
            output['texts'].append(text)
            output['tags'].append(tag)
            # output['tags'].append([0]+tag if text[0]!='' else tag) # [0]+tag so that in all case, the first tag refers to [CLS]
            # not necessary since all the leading punctuations are stripped
        return output
    return chunk_examples
assert(chunk_examples_with_degree(0,{'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, '?': 6, '—': 7},{'…':'.',';':'.'})(['Hello!Bye…'])=={'texts': [['Hello', 'Bye']], 'tags': [[1, 4]]})

def subword_tokenize(tokenizer,tokens, pad_start):
    '''convert word list into list of subword ids'''
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = list(flatten(subwords))
    token_start_idxs = np.cumsum([pad_start]+subword_lengths[:-1])
    token_end_idxs = np.cumsum([pad_start]+subword_lengths[:-1])+np.array(subword_lengths)-1
    return ["[PAD]"]*pad_start+subwords, token_start_idxs, token_end_idxs

def chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,pad_start:int,tokens, labels=None):
    subwords,token_start_idxs,token_end_idxs = subword_tokenize(tokenizer,tokens, pad_start)
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
    
def chunk_to_len_batch(max_seq_length,
    tokenizer,
    tokens,
    labels=None,
    labelled=True,
    ignore_index=-100, 
    attach_label_to_end=None,
    no_space_label=None,
    pad_start=0):
    no_mask=False
    if attach_label_to_end is None:
        no_mask=True
        attach_label_to_end=True
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    for i,_ in enumerate(zip(tokens,tokens) if labels==None else zip(tokens,labels)):
        a,b,c=chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,pad_start,*_) if labels else chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,pad_start,_[0])
        
        batch_ids.extend(a[:min(len(a),len(b))])
        batch_masks.extend(b[:min(len(a),len(b))])
        if labelled==True:
            batch_labels.extend(c)
    output = {'input_ids': torch.as_tensor(batch_ids, dtype=torch.long),
              'attention_mask': torch.as_tensor(batch_ids, dtype=torch.bool),
              'subtoken_mask': torch.as_tensor(batch_masks,dtype=torch.bool)}
    if labelled==True:
        output['labels']=torch.as_tensor(batch_labels,dtype=torch.long)
    else:
        output['labels']=torch.zeros_like(output['input_ids'],dtype=torch.long)
    if no_mask:
        if (labelled==True and attach_label_to_end and no_space_label is not None):
            # no_space_label is 2
            output['labels']+=no_space_label*((output['attention_mask']^output['subtoken_mask'])*(output['input_ids']!=102)*(output['input_ids']!=101))
        output['subtoken_mask']=output['attention_mask']&(output['input_ids']!=102)
    else:
        output['subtoken_mask']|=(output['input_ids']==101)  # dont want end token |(output['input_ids']==102)
        output['subtoken_mask']&=labelled
    
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
#%%
import regex as re
import random

def sent_tokenize(text):
    '''Split at sentence boundaries (using my own set of sentence boundary punctuation)'''
    return re.findall(r"\w.{20,}?[.!?… ]*[.!?…] |\w.+[.!?… ]*[.!?…]$",text)


def shuffle_sentence_transform(data, always_apply=False, p=0.5):
    '''swap sentences within example'''
    text = data
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    return ' '.join(sentences)

def swap_transform(data, distance=1, probability=0.1, always_apply=False, p=0.5):
    '''swap adjacent words'''
    swap_range_list = list(range(1, distance+1))
    text = data
    words = text.split()
    words_count = len(words)
    if words_count <= 1:
        return text

    new_words = {}
    for i in range(words_count):
        if random.random() > probability:
            new_words[i] = words[i]
            continue

        if i < distance:
            new_words[i] = words[i]
            continue

        swap_idx = i - random.choice(swap_range_list)
        new_words[i] = new_words[swap_idx]
        new_words[swap_idx] = words[i]

    return ' '.join([v for k, v in sorted(new_words.items(), key=lambda x: x[0])])

def delete_transform(data, probability=0.05, always_apply=False, p=0.5):
    '''delete random words'''
    text = data
    words = text.split()
    words_count = len(words)
    if words_count <= 1:
        return text
    
    new_words = []
    for i in range(words_count):
        if (random.random() < probability and words[i].isalpha()):
            continue
        new_words.append(words[i])

    if len(new_words) == 0:
        return words[random.randint(0, words_count-1)]

    return ' '.join(new_words)

def substitute_transform(data, tokenizer,probability=0.05, always_apply=False, p=0.5):
    '''substitute random words randomly'''
    text = data
    words = text.split()
    words_count = len(words)
    if words_count <= 1:
        return text
    
    new_words = []
    for i in range(words_count):
        if (random.random() < probability and words[i].isalpha()):
            randtoken=tokenizer.convert_ids_to_tokens(random.randint(0,tokenizer.vocab_size))
            if randtoken is None:
                new_words.append('[UNK]')
            elif randtoken[0].isalpha():
                new_words.append(randtoken)
            else:
                new_words.append('[UNK]')
            continue
        new_words.append(words[i])

    if len(new_words) == 0:
        return words[random.randint(0, words_count-1)]

    return ' '.join(new_words)

def insert_transform(data, tokenizer, probability=0.1, always_apply=False, p=0.5):
    '''insert random words'''
    text = data
    words = text.split()
    words_count = len(words)
    if words_count <= 1:
        return text
    
    new_words = []
    for i in range(words_count):
        if (random.random() < probability and words[i].isalpha()):
            randtoken=tokenizer.convert_ids_to_tokens(random.randint(0,tokenizer.vocab_size))
            if randtoken is None:
                new_words.append('[UNK]')
            elif randtoken[0].isalpha():
                new_words.append(randtoken)
            else:
                new_words.append('[UNK]')
        new_words.append(words[i])

    if len(new_words) == 0:
        return words[random.randint(0, words_count-1)]

    return ' '.join(new_words)

def split_transform(data, probability=0.05, always_apply=False, p=0.5):
    '''split words randomly'''
    text = data
    words = text.split()
    words_count = len(words)
    if words_count <= 1:
        return text
    
    new_words = []
    for i in range(words_count):
        if (random.random() < probability and words[i].isalpha() and len(words[i])>3):
            x=random.randint(1,len(words[i])-1)
            new_words.append(words[i][:x])
            new_words.append(words[i][x:])
        else:
            new_words.append(words[i])

    if len(new_words) == 0:
        return words[random.randint(0, words_count-1)]

    return ' '.join(new_words)


def all_transform(text,tokenizer,alpha_sub=0.4, alpha_del=0.4, alpha_ins=0.4,alpha_swp=0, alpha_split=0.1):
    '''wrapper for all defined random transformations. Have to reduce and fix the maximum probability of transform.'''  
    r=np.random.rand(5)
    text=shuffle_sentence_transform(text)
    if r[0] < alpha_sub:
        substitute_transform(text,tokenizer)
    if r[1] < alpha_del:
        text=delete_transform(text)
    if r[2] < alpha_ins:
        text=insert_transform(text,tokenizer)
    if r[3] < alpha_swp:
        text=swap_transform(text)
    if r[4] < alpha_split:
        text=split_transform(text)
    return text
