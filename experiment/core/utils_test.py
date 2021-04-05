from core.utils import *
# from utils import *
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
# import pytorch_lightning as pl
__all__ = ['test_mask']

# pl.seed_everything(0)
def test_mask(masktype,sigma,tokenizer,labels_to_ids,label_map,max_seq_length,stride,attach_label_to_end,no_space_label,labelled=True):
    ids_to_labels={v:k for k,v in labels_to_ids.items()}
    chunk0=chunk_examples_with_degree(0,labels_to_ids,label_map)
    sample1=chunk0(["Itttt can be a very complicated thing, the ocean. ","And it can be a very complicated thing, what human health is. And bringing those two together might seem a very daunting task, but what I'm going to try to say is that even in that complexity, there's some simple themes that I think, if we understand, we can really move forward. And those simple themes aren't really themes about the complex science of what's going on, but things that we all pretty well know. And I'm going to start with this one: If momma ain't happy, ain't nobody happy. We know that, right? We've experienced that. And if we just take that and we build from there, then we can go to the next step, which is that if the ocean ain't happy, ain't nobody happy. That's the theme of my talk. And we're making the ocean pretty unhappy in a lot of different ways. This is a shot of Cannery Row in 1932. Cannery Row, at the time, had the biggest industrial canning operation on the west coast. We piled enormous amounts of pollution into the air and into the water. Rolf Bolin, who was a professor at the Hopkin's Marine Station where I work, wrote in the 1940s that The fumes from the scum floating on the inlets of the bay were so bad they turned lead-based paints black. People working in these canneries could barely stay there all day because of the smell, but you know what they came out saying? They say, You know what you smell? You smell money. That pollution was money to that community, and those people dealt with the pollution and absorbed it into their skin and into their bodies because they needed the moneyyyyyy."])
    texts=sample1['texts']
    tags=sample1['tags']
    batched=chunk_to_len_batch(max_seq_length,tokenizer,texts,tags,labelled,attach_label_to_end=attach_label_to_end,no_space_label=labels_to_ids[no_space_label], stride=stride)
    for i,j in zip(batched['labels'][:-1],batched['labels'][1:]):
        assert((i[stride*1+1:stride*2+1]==j[stride*0+1:stride*1+1]).all(),(i[stride*1+1:stride*2+1]-j[stride*0+1:stride*1+1]))
    mask=get_mask(masktype,max_seq_length-2,sigma)
    print('mask:',mask)
    num_classes=len(labels_to_ids)
    result,label,combinedmask=combine_preds(F.one_hot(batched['labels'],num_classes),batched['input_ids'],batched['subtoken_mask'],mask.unsqueeze(-1),stride,batched['labels'],num_classes)
    print((torch.argmax(result[combinedmask],dim=1)-label[combinedmask]))
    assert((torch.argmax(result[combinedmask],dim=1)==label[combinedmask]).all())
    return mask
# tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')
# labels_to_ids={'': 0, '!': 1, ',': 2, '-': 3, '.': 4, ':': 5, '?': 6, '—': 7}
# ids_to_labels={v:k for k,v in labels_to_ids.items()}
# label_map={'…':'.',';':'.'}
# max_seq_length=128 ## ensure of the form 2^(odd integer)
# stride=126 ## (1/3 * )

# test_mask('uniform',1,tokenizer,labels_to_ids, label_map, max_seq_length,stride,True,no_space_label='')

# if max_seq_length is of the form 2^(odd integer): supports 1/2 and 1/3 stride
# if max_seq_length is of the form 2^(even integer): supports 1/2 stride


# chunk0=chunk_examples_with_degree(0,labels_to_ids,label_map)
# assert (chunk0(['Hello!Bye…'])=={'texts': [['Hello', 'Bye']], 'tags': [[1, 4]]})
# sample1=chunk0(["Itttt can be a very complicated thing, the ocean. ","And it can be a very complicated thing, what human health is. And bringing those two together might seem a very daunting task, but what I'm going to try to say is that even in that complexity, there's some simple themes that I think, if we understand, we can really move forward. And those simple themes aren't really themes about the complex science of what's going on, but things that we all pretty well know. And I'm going to start with this one: If momma ain't happy, ain't nobody happy. We know that, right? We've experienced that. And if we just take that and we build from there, then we can go to the next step, which is that if the ocean ain't happy, ain't nobody happy. That's the theme of my talk. And we're making the ocean pretty unhappy in a lot of different ways. This is a shot of Cannery Row in 1932. Cannery Row, at the time, had the biggest industrial canning operation on the west coast. We piled enormous amounts of pollution into the air and into the water. Rolf Bolin, who was a professor at the Hopkin's Marine Station where I work, wrote in the 1940s that The fumes from the scum floating on the inlets of the bay were so bad they turned lead-based paints black. People working in these canneries could barely stay there all day because of the smell, but you know what they came out saying? They say, You know what you smell? You smell money. That pollution was money to that community, and those people dealt with the pollution and absorbed it into their skin and into their bodies because they needed the moneyyyyyy."])
# texts=sample1['texts']
# tags=sample1['tags']
# print(texts,tags,len(tags))
# batched=chunk_to_len_batch(max_seq_length,tokenizer,texts,tags,True,attach_label_to_end=None,no_space_label=0, stride=stride)
# for i,j in zip(batched['labels'][:-1],batched['labels'][1:]):
#     assert((i[stride*1+1:stride*2+1]==j[stride*0+1:stride*1+1]).all())

# # print(all_transform("Itttt can be a very complicated thing, the ocean. And it can be a very complicated thing, what human health is. And bringing those two together might seem a very daunting task, but what I'm going to try to say is that even in that complexity, there's some simple themes that I think, if we understand, we can really move forward. And those simple themes aren't really themes about the complex science of what's going on, but things that we all pretty well know. And I'm going to start with this one: If momma ain't happy, ain't nobody happy. We know that, right? We've experienced that. And if we just take that and we build from there, then we can go to the next step, which is that if the ocean ain't happy, ain't nobody happy. That's the theme of my talk. And we're making the ocean pretty unhappy in a lot of different ways. This is a shot of Cannery Row in 1932. Cannery Row, at the time, had the biggest industrial canning operation on the west coast. We piled enormous amounts of pollution into the air and into the water. Rolf Bolin, who was a professor at the Hopkin's Marine Station where I work, wrote in the 1940s that The fumes from the scum floating on the inlets of the bay were so bad they turned lead-based paints black. People working in these canneries could barely stay there all day because of the smell, but you know what they came out saying? They say, You know what you smell? You smell money. That pollution was money to that community, and those people dealt with the pollution and absorbed it into their skin and into their bodies because they needed the moneyyyyyy.", tokenizer))
# '''
# def combinepreds(preds,input_ids,subtoken_mask,mask,stride,labels=None,num_labels=8):
#     if stride==0:
#         stride=len(mask)
#     combined_mask=torch.zeros((len(preds)-1)*stride+len(mask))
#     # print((len(preds)-1),stride,len(mask),(len(preds)-1)*stride+mask.shape[0],len(combined_mask))
#     combined_result=torch.zeros(len(combined_mask),num_labels)
#     combined_labels=torch.zeros(len(combined_mask)) if labels is not None else None
#     # print('crshape',combined_result.shape)
#     offset=0
#     for i in range(len(preds)):
#         # print(torch.argmax(preds[i][1:-1],dim=1))#,combined_result[offset:offset+len(mask)].shape)
#         # print(subtoken_mask[i])
#         combined_result[offset:offset+len(mask)]+=preds[i][1:-1]*mask
#         if labels is not None:
#             combined_labels[offset:offset+len(mask)]=labels[i][1:-1]
#         combined_mask[offset:offset+len(mask)]=subtoken_mask[i][1:-1]
#         # new=(previous[stride+1:-1]==current[1:-stride-1]).all()
#         # print(pred,previous[stride+1:-1],current[1:-1],new,'\n')#)
#         offset+=stride
#         # print(offset)
#     # return preds
#     return combined_result,combined_labels,combined_mask.bool()
# '''

# mask=get_mask('uniform',126,3)
# print(mask,sum(mask),sum(mask[42:84]))

# num_classes=len(labels_to_ids)
# print(F.one_hot(batched['labels'],num_classes).shape)
# result,label,combinedmask=combine_preds(F.one_hot(batched['labels'],num_classes),batched['input_ids'],batched['subtoken_mask'],mask.unsqueeze(-1),stride,batched['labels'],num_classes)
# print((torch.argmax(result[combinedmask],dim=1)==label[combinedmask]).all())
# # print(view_aligned(batched['input_ids'],batched['labels'],tokenizer,ids_to_labels))


# # %%
