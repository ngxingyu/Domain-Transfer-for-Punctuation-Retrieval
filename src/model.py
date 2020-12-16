from config import *
import torch
import transformers
from torch import nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_punct):
        super(EntityModel, self).__init__()
        self.num_punct = num_punct
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH
        )
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_punct = nn.Linear(768, self.num_punct)

    def forward(
        self,
        data,
        #ids,
        #mask,
        #token_type_ids,
        #target_punct,
    ):
        o1 = self.bert(
            data[0],#ids,
            attention_mask=data[1],#mask,
            #token_type_ids=token_type_ids
        )[0]
        bo_punct = self.bert_drop_1(o1)

        punct = self.out_punct(bo_punct)

        loss_punct = loss_fn(punct, data[2], data[1], self.num_punct)

        loss = loss_punct
        #loss = (loss_tag + loss_pos) / 2

        return punct, loss
