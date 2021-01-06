from .config import *
import torch
import transformers
from torch import nn
import torch.nn.functional as F
from typing import Optional
from .CRF import *

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
                input: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                num_classes: int = 10,
                ) -> torch.Tensor:
        input_soft = torch.softmax(input[target!=-100],dim=1)
        target_one_hot=F.one_hot(target[target!=-100],num_classes=num_classes)
        input_factor = ((1-input_soft) ** self.alpha) if self.self_adjusting else 1
        if mask is not None:
            mask = mask.view(-1).float()
            input_soft = input_soft * mask
            target_one_hot = target_one_hot * mask
        intersection = torch.sum(input_factor*input_soft * target_one_hot, 0)
        cardinality = torch.sum(input_factor*torch.square(input_soft,) + torch.square(target_one_hot,), 0) if self.square_denominator else torch.sum(input_factor*input_soft + target_one_hot, 0)
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

def loss_fn(output, target, mask, weight=None):
    lfn = DiceLoss(square_denominator=config.square_denominator,self_adjusting=config.self_adjusting,alpha=config.alpha,weight=weight)
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, config.num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(-100).type_as(target)
    )
    loss = lfn(active_logits, active_labels,num_classes=config.num_labels)
    return loss


class BertCRFModel(nn.Module):
    def __init__(self,num_punct, embedding_dim, hidden_dim, use_lstm=False, use_crf=True, logger=None):
        super(BertCRFModel, self).__init__()
        self.num_punct=num_punct
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        #self.use_lstm = use_lstm
        self.use_crf = use_crf
        self.logger=logger
        self.bert = transformers.BertModel.from_pretrained(
            config.base_model_path
        )
        self.bert_drop_1 = nn.Dropout(config.hidden_dropout_prob)
        #if self.use_lstm:
        #    self.lstm=nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)
        #    self.out_punct = nn.Linear(self.hidden_dim, self.num_punct)
        #else:
        self.out_punct = nn.Linear(self.embedding_dim, self.num_punct)
        if self.use_crf:
            self.crf= DiceCRF(self.num_punct)

    def forward(self, data):
        o1 = self.bert(
                data[0],				#ids
                attention_mask=data[1],	#mask,
        )[0]
        sequence_output = self.bert_drop_1(o1)
        #self.logger.info('bert output shape: {}'.format(sequence_output.shape))
        #if self.use_lstm:
        #    sequence_output=self.lstm(sequence_output)[0]
        #    self.logger.info('lstm output shape: {}'.format(sequence_output.shape))
        punct = self.out_punct(sequence_output)
        #self.logger.info('punct shape: {}'.format(punct.shape))
        if self.use_crf:
            loss= -1*self.crf(punct, data[2], data[1])
        else:
            loss = loss_fn(punct, data[2], data[1], self.num_punct,1)
        #loss = (loss_tag + loss_pos) / 2

        return punct, loss