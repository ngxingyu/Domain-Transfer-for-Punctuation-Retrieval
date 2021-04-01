#%% 
from utils import *
from transformers import AutoTokenizer
import torch
# if max_seq_length is of the form 2^(odd integer): supports 1/2 and 1/3 stride
# if max_seq_length is of the form 2^(even integer): supports 1/2 stride

tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
labels_to_ids={'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, '?': 6, '—': 7}
ids_to_labels={v:k for k,v in labels_to_ids.items()}
label_map={'…':'.',';':'.'}
max_seq_length=128 ## ensure of the form 2^(odd integer)
stride=42 ## (1/3 * )
chunk0=chunk_examples_with_degree(0,labels_to_ids,label_map)
assert (chunk0(['Hello!Bye…'])=={'texts': [['Hello', 'Bye']], 'tags': [[1, 4]]})
sample1=chunk0(["Itttt can be a very complicated thing, the ocean. ","And it can be a very complicated thing, what human health is. And bringing those two together might seem a very daunting task, but what I'm going to try to say is that even in that complexity, there's some simple themes that I think, if we understand, we can really move forward. And those simple themes aren't really themes about the complex science of what's going on, but things that we all pretty well know. And I'm going to start with this one: If momma ain't happy, ain't nobody happy. We know that, right? We've experienced that. And if we just take that and we build from there, then we can go to the next step, which is that if the ocean ain't happy, ain't nobody happy. That's the theme of my talk. And we're making the ocean pretty unhappy in a lot of different ways. This is a shot of Cannery Row in 1932. Cannery Row, at the time, had the biggest industrial canning operation on the west coast. We piled enormous amounts of pollution into the air and into the water. Rolf Bolin, who was a professor at the Hopkin's Marine Station where I work, wrote in the 1940s that The fumes from the scum floating on the inlets of the bay were so bad they turned lead-based paints black. People working in these canneries could barely stay there all day because of the smell, but you know what they came out saying? They say, You know what you smell? You smell money. That pollution was money to that community, and those people dealt with the pollution and absorbed it into their skin and into their bodies because they needed the moneyyyyyy."])
texts=sample1['texts']
tags=sample1['tags']
print(texts,tags,len(tags))
batched=chunk_to_len_batch(max_seq_length,tokenizer,texts,tags,True,attach_label_to_end=None,no_space_label=0, stride=stride)
for i,j in zip(batched['labels'][:-1],batched['labels'][1:]):
    assert((i[stride*1+1:stride*2+1]==j[stride*0+1:stride*1+1]).all())

# print(all_transform("Itttt can be a very complicated thing, the ocean. And it can be a very complicated thing, what human health is. And bringing those two together might seem a very daunting task, but what I'm going to try to say is that even in that complexity, there's some simple themes that I think, if we understand, we can really move forward. And those simple themes aren't really themes about the complex science of what's going on, but things that we all pretty well know. And I'm going to start with this one: If momma ain't happy, ain't nobody happy. We know that, right? We've experienced that. And if we just take that and we build from there, then we can go to the next step, which is that if the ocean ain't happy, ain't nobody happy. That's the theme of my talk. And we're making the ocean pretty unhappy in a lot of different ways. This is a shot of Cannery Row in 1932. Cannery Row, at the time, had the biggest industrial canning operation on the west coast. We piled enormous amounts of pollution into the air and into the water. Rolf Bolin, who was a professor at the Hopkin's Marine Station where I work, wrote in the 1940s that The fumes from the scum floating on the inlets of the bay were so bad they turned lead-based paints black. People working in these canneries could barely stay there all day because of the smell, but you know what they came out saying? They say, You know what you smell? You smell money. That pollution was money to that community, and those people dealt with the pollution and absorbed it into their skin and into their bodies because they needed the moneyyyyyy.", tokenizer))
def combinepreds(preds,input_ids,mask,stride):
    previous=torch.zeros_like(mask)
    for pred,current in zip(preds,input_ids):
        new=(previous[stride+1:-1]==current[1:-stride-1]).all()
        print(pred,previous[stride+1:-1],current[1:-1],new,'\n')#)
        previous=current
    return preds

mask=torch.zeros(max_seq_length)
mask[(max_seq_length-2)//3:(max_seq_length-2)//3*2]=1
combinepreds(batched['labels'],batched['input_ids'],mask,stride)
# print(view_aligned(batched['input_ids'],batched['labels'],tokenizer,ids_to_labels))


# %%
