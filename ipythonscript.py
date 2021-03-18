#%autoindent
#%%
import torch
from torch import nn
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor
from typing import Optional
from datasets import load_dataset
import transformers
from transformers import DistilBertTokenizerFast, BertPreTrainedModel, get_linear_schedule_with_warmup, AdamW
from torchcontrib.optim import SWA
import regex as re
import numpy as np
from tqdm import tqdm

#%%
ted=load_dataset('csv',data_files={'train':'/home/nxingyu/project/data/ted_talks_processed.train.csv',
                                   'dev':'/home/nxingyu/project/data/ted_talks_processed.dev.csv',
                                   'test':'/home/nxingyu/project/data/ted_talks_processed.test.csv'})


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
tags=sorted(list('.?!,;:-—…'))
tag2id = {tag: id+1 for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
tag2id,id2tag

class PunctuationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids:LongTensor, attention_mask:LongTensor, labels:Optional[LongTensor] = None) -> None:
        """
        :param input_ids: tokenids
        :param attention_mask: attention_mask, null->0
        :param labels: true labels, optional
        :return None
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __getitem__(self, idx):
        #Is torch.as_tensor neccessary?
        """:param idx: implement index"""
        item = {'input_ids': torch.as_tensor(self.input_ids[idx],dtype=torch.long),
        'attention_mask': torch.as_tensor(self.attention_mask[idx],dtype=torch.long),
        'labels': torch.as_tensor(self.labels[idx],dtype=torch.long)
        }
        return item
    def __len__(self):
        return len(self.labels)



tags=sorted(list('.?!,;:-—…'))
tag2id = {tag: id+1 for id, tag in enumerate(tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
'''
#processing
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
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
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
    # return torch.utils.data.TensorDataset({'input_ids':torch.tensor(encodings['input_ids'],dtype=torch.long),
    # 'attention_mask':torch.tensor(encodings['attention_mask'],dtype=torch.long),
    # 'labels':torch.tensor(labels,dtype=torch.long)})
    return PunctuationDataset(torch.tensor(encodings['input_ids'],dtype=torch.long),
        torch.tensor(encodings['attention_mask'],dtype=torch.long),
        torch.tensor(labels,dtype=torch.long))
test_dataset=process_dataset(ted,'test')#,10,3)
dev_dataset=process_dataset(ted,'dev')
train_dataset=process_dataset(ted,'train')

for name,dataset in {'test':test_dataset,'train':train_dataset, 'dev':dev_dataset, }.items():# 'train':train_dataset, 'dev':dev_dataset,
    torch.save(dataset, './ted-'+name+'.pt')

test_dataset=torch.load('~/project/data/ted-test.pt')
test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=4)
'''

config = transformers.configuration_utils.PretrainedConfig.from_dict(
{   "max_len" : 128,
    "overlap" : 126,
    "train_batch_size" : 64,
    "dev_batch_size" : 64,
    "gpu_device" : 'cuda:0',#'cpu'#
    "freeze_epochs" : 20,
    "freeze_lr" : 1e-4,
    "unfreeze_epochs" : 20,
    "unfreeze_layers" : 6,
    "unfreeze_lr" : 1e-5,
    "base_model_path" : 'distilbert-base-uncased',
    "train_dataset" : '/home/nxingyu/project/data/ted_talks_processed.dev.pt',
    "dev_dataset" : '/home/nxingyu/project/data/ted_talks_processed.dev.pt',
    "alpha" : 0.8,
    "hidden_dropout_prob" : 0.3,
    "embedding_dim" : 768,
    "num_labels" : 10,
    "hidden_dim" : 128,
    "self_adjusting":True,
    "square_denominator":False,
    "use_crf":True,
    "model_name" : 'bertcrf',
    "model_path" : "/home/nxingyu/project/logs/models/"})

#check if map_location=config.device works
train_dataset=PunctuationDataset(*torch.load(config.train_dataset,map_location=config.device)[:])
dev_dataset=PunctuationDataset(*torch.load(config.dev_dataset,map_location=config.device)[:])
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=4)
dev_dataloader=torch.utils.data.DataLoader(dev_dataset, batch_size=config.dev_batch_size, num_workers=2)



class DiceLoss(nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 self_adjusting: Optional[bool] = False,
                #  with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean",
                 alpha: float = 1.0,
                 ignore_index: int = -100,
                 weight=1,
                 ) -> None:
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.self_adjusting = self_adjusting
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.weight=weight
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                num_classes: int = 10,
                ) -> torch.Tensor:
        pred_soft = torch.softmax(pred[target!=-100],dim=1) #(batch_size,seq_len,num_labels)->(batch_size,seq_len,num_labels)
        target_one_hot=F.one_hot(target[target!=-100],num_classes=num_classes) #(batch_size,seq_len,1)->(batch_size,seq_len,num_labels)
        pred_factor = ((1-pred_soft) ** self.alpha) if self.self_adjusting else 1
        if mask is not None:
            mask = mask.view(-1).float()
            pred_soft = pred_soft * mask
            target_one_hot = target_one_hot * mask
        intersection = torch.sum(pred_factor*pred_soft * target_one_hot, 0)
        cardinality = torch.sum(pred_factor*torch.square(pred_soft,) + torch.square(target_one_hot,), 0) if self.square_denominator else torch.sum(pred_factor*pred_soft + target_one_hot, 0)
#         assert 0,f'intersection {intersection} cardinality {cardinality} smooth {self.smooth} weight {self.weight}'
        #intersection tensor([4.5844e+02, 1.2329e-01, 4.1284e+01, 3.2318e+00, 2.8283e+01, 6.0894e-01, 4.0947e-01, 1.6453e+00, 2.8845e+00, 0.0000e+00],
       #grad_fn=<SumBackward1>) cardinality tensor([7014.1035,  639.7080, 1117.3043, 1100.2483,  958.1732,  472.2272, 840.4181,  668.1088,  732.5102,  632.7620], grad_fn=<SumBackward1>) smooth 1e-08 weight 1

        dice_score = 1. - 2. * intersection / (cardinality + self.smooth) * self.weight
        if self.reduction == "mean":
            return dice_score.mean()
        elif self.reduction == "sum":
            return dice_score.sum()
        elif self.reduction == "none" or self.reduction is None:
            return dice_score
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}"

def dice_loss(output, target, mask, weight=1):
    lfn = DiceLoss(square_denominator=config.square_denominator,self_adjusting=config.self_adjusting,alpha=config.alpha, weight=weight)
    active_loss = mask.view(-1) == 1 # (batch_size,seq_len)->(batch_size*seq_len)
    active_logits = output.view(-1, config.num_labels) # (batch_size, seq_len ,num_labels) -> (batch_size*seq_len,num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(-100).type_as(target)
    ) # -100 if out of mask else target (batch_size*seq_len)
#     assert 0, f'{target.size()},{active_labels.size()}'
    loss = lfn(active_logits, active_labels,num_classes=config.num_labels)
    print('loss',loss)
    return loss


class BertCRFModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.num_labels=config.num_labels
        self.embedding_dim=config.embedding_dim
        self.hidden_dim=config.hidden_dim
        self.use_crf = config.use_crf
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.logger=config.logger
        self.bert = transformers.BertModel.from_pretrained(
            config.base_model_path
        )
        self.fcl = nn.Linear(self.embedding_dim, self.num_labels)
        if self.use_crf:
            self.crf= DiceCRF(self.num_labels)
        else:
            self.dice_loss=dice_loss
    def forward(self, input_ids:LongTensor, attn_masks:LongTensor, labels:Optional[LongTensor]=None):
        """
        :param input_ids: tokenids (batch_size,seq_len)
        :param attention_mask: attention_mask, null->0 (batch_size,seq_len)
        :param labels: true labels, optional (batch_size,seq_len)
        :return None
        """
        o1 = self.bert(input_ids,attn_masks)[0] #(batch_size,seq_len), (batch_size,seq_len)-> last_hidden_state (batch_size,seq_len,hidden_size)
        sequence_output = self.dropout(o1) #(batch_size,seq_len,hidden_size)
        punct = self.fcl(sequence_output) #(batch_size,seq_len,hidden_size) -> (batch_size,seq_len,num_labels)
        if self.use_crf:
            if labels is not None:
                loss = -self.crf(punct, labels, mask=attn_masks, reduction='sum')
                return loss
            else:
                prediction = self.crf.viterbi_decode(punct, attn_masks)
                return prediction
        else:
            if labels is not None:
                loss = self.dice_loss(punct, labels, attn_masks)
                return loss
            else:
#                 batch_size, seq_len, _ = punct.size()
#                 print(punct)
                return punct.argmax(-1)
        #loss = (loss_tag + loss_pos) / 2



model= BertCRFModel(config)
device = torch.device(config.gpu_device) if torch.cuda.is_available() else torch.device('cpu')
for i,param in enumerate(model.bert.parameters()):
    param.requires_grad = False
model.to(device)

# for batch in train_dataloader:
#     print(model(batch['input_ids'],batch['attention_mask'],labels=batch['labels']))


param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = train_dataset.labels[0].size()[0] / config.train_batch_size * config.freeze_epochs

base_opt = AdamW(optimizer_parameters, lr=config.freeze_lr)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids, attention_mask, labels=labels)
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids, attention_mask, labels=labels)
        print(loss)
        final_loss += loss.item()
    return final_loss / len(data_loader)


for epoch in range(config.freeze_epochs):
    train_loss = train_fn(
        train_dataloader,
        model,
        optimizer,
        device,
        scheduler
    )
    optimizer.update_swa()
    optimizer.swap_swa_sgd()
    test_loss = eval_fn(
        dev_dataloader,
        model,
        device
    )
