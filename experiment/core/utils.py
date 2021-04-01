import numpy as np
import torch
from torch import nn
import regex as re
import snoop
from copy import deepcopy
import itertools
from scipy.stats import norm,trapezoid,uniform

__all__ = ['chunk_examples_with_degree', 'chunk_to_len_batch', 'flatten', 'chunk_to_len', 'view_aligned', 'all_transform','get_mask','combine_preds']

def get_mask(type='normal',max_seq_length=126,sigma=10):
    '''normal, trapezoid, uniform'''
    if type=='normal':
        dist= norm.pdf(torch.arange(-max_seq_length/2,max_seq_length/2,1),0,max_seq_length/sigma)
    elif type=='trapezoid':
        dist= trapezoid.pdf(torch.arange(-max_seq_length/2,max_seq_length/2,1),0.5-1/(2*sigma),0.5+1/(2*sigma),-max_seq_length/2,max_seq_length)
    elif type=='uniform':
        dist= uniform.pdf(torch.arange(-max_seq_length/2,max_seq_length/2,1),-max_seq_length/(2*sigma),max_seq_length/sigma)
    return torch.tensor(dist*max_seq_length/sum(dist))

def combine_preds(preds,input_ids,subtoken_mask,mask,stride,labels=None,num_labels=8):
    '''merge overlapping entries with stride'''
    if stride==0:
        stride=len(mask)
    combined_mask=torch.zeros((len(preds)-1)*stride+len(mask)).type_as(mask)
    combined_result=torch.zeros(len(combined_mask),num_labels).type_as(mask)
    combined_labels=torch.zeros(len(combined_mask)).type_as(input_ids) if labels is not None else None
    # pp(combined_mask.type(),combined_result.type(),combined_labels.type())
    offset=0
    for i in range(len(preds)):
        combined_result[offset:offset+len(mask)]+=preds[i][1:-1]*mask
        if labels is not None:
            combined_labels[offset:offset+len(mask)]=labels[i][1:-1]
        combined_mask[offset:offset+len(mask)]=subtoken_mask[i][1:-1]
        offset+=stride
    return combined_result,combined_labels.long(),combined_mask.bool()

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

def view_aligned(texts,tags,tokenizer,ids_to_labels):
    '''Convert tokens ids and labels into readable text'''
    output=[re.sub(r'( ?\[((CLS))\] ?)',' ',
        re.sub(" \' ","\'",
            re.sub('# +','',
                re.sub('#? +##','',
                    ' '.join( #[.?!,;:\-—… ]+
                        [_[0]+_[1] for _ in list(
                            zip(tokenizer.convert_ids_to_tokens(_[0]),
                                [ids_to_labels[id] for id in _[1].tolist()])
                            )
                        ]
                    )
                )
            )
        )
    ) for _ in zip(texts,tags)]
    newoutput=[]
    prevappend=False
    # print(len(output))
    for value in output:
        if value[:5]!='[PAD]':
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
    padding=np.arange(pad_start)
    endpad=(token_end_idxs[-1]+1)*np.ones_like(padding)
    return ["[PAD]"]*pad_start+subwords+["[PAD]"]*pad_start, np.concatenate([padding,token_start_idxs]), np.concatenate([padding,token_end_idxs])

def trim(filt, trim_value='[PAD]', trim='fb'):
    """
    Copied from np.trim_zeros, but allows the trim value to be specified
    """
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if i != trim_value:
                break
            else:
                first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if i != trim_value:
                break
            else:
                last = last - 1
    return filt[first:last]

def mergelists(l):
    return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*l)) if x is not None]

def chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,stride, tokens, labels=None):
    if (stride==None) or (stride==0):
        stride=max_seq_length-2
    pad_start=max_seq_length-2-stride
    assert((max_seq_length-2)%stride==0)
    numstrides=(max_seq_length-2)//stride
    subwords,token_start_idxs,token_end_idxs = subword_tokenize(tokenizer,tokens, pad_start)
    token_idxs=token_end_idxs if attach_label_to_end else token_start_idxs
    labels=[0]*(max_seq_length-2-stride)+labels#+[0]*(max_seq_length-2-stride) # pad start and end of labels so first example only sees stride tokens.
    teim=token_idxs%(stride)
    stridecount=token_idxs//stride
    # breakpoints=(np.argwhere(stridecount[1:]>stridecount[:-1]).flatten()+1).tolist()
    # split_token_idxs=np.array_split(token_idxs,breakpoints)
    # assert(sum(len(_) for _ in split_token_idxs)==len(labels))
    # print([np.split(subwords,np.arange(stride*x,len(labels),max_seq_length-2)) for x in range(1,1+numstrides)])
    split_subwords=mergelists([np.split(subwords,np.arange(stride*x,len(subwords),max_seq_length-2)) for x in range(1,1+numstrides)])[numstrides-1:]
    # split_subwords=np.array_split(subwords,np.arange(max_seq_length-2,len(subwords),max_seq_length-2))
    ids=[pad_to_len(max_seq_length,tokenizer.convert_tokens_to_ids(['[CLS]']+list(trim(_,'[PAD]','b'))+['[SEP]'])) for _ in split_subwords]
    # print(token_idxs,split_subwords)
    stridecount=[(token_idxs-stride*i)//(max_seq_length-2) for i in range(1,1+numstrides)]
    breakpoints=[(np.argwhere(_[1:]>_[:-1]).flatten()+1).tolist() for _ in stridecount]
    # print([len(x) for x in breakpoints],breakpoints,stridecount)
    split_token_idxs=mergelists([np.array_split((token_idxs-(i+1)*stride)%(max_seq_length-2),v) for i,v in enumerate(breakpoints)])[numstrides-1:]
    split_subwords=split_subwords[:len(split_token_idxs)]
    # print(len(split_subwords),len(split_token_idxs))
    # print(mergelists([split_token_idxs,split_subwords]))
    assert(len(split_subwords)==len(split_token_idxs))

    masks=[]
    for _ in split_token_idxs:
        masks.append(position_to_mask(max_seq_length,_).copy())

    # print(list(zip(split_subwords,masks))[0])
    assert(len(masks)==len(split_subwords))
    padded_labels=None
    if labels!=None:
        split_labels=mergelists([np.array_split(labels,v) for v in breakpoints])[numstrides-1:]
        assert(len(split_labels)==len(split_subwords))
        padded_labels=[pad_to_len(max_seq_length,align_labels_to_mask(*_)) for _ in zip(masks,split_labels)]
    # print(list(zip(*[ids,masks,padded_labels])))
    return ids,masks,padded_labels
    
def chunk_to_len_batch(max_seq_length,
    tokenizer,
    tokens,
    labels=None,
    labelled=True,
    ignore_index=-100, 
    attach_label_to_end=None,
    no_space_label=None,
    stride=None):
    no_mask=False
    if attach_label_to_end is None:
        no_mask=True
        attach_label_to_end=True
    batch_ids=[]
    batch_masks=[]
    batch_labels=[]
    for i,_ in enumerate(zip(tokens,tokens) if labels==None else zip(tokens,labels)):
        a,b,c=chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,stride,*_) if labels else chunk_to_len(max_seq_length,tokenizer,attach_label_to_end,stride,_[0])
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
    # print(torch.logical_xor(output['attention_mask'],output['subtoken_mask']))
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
