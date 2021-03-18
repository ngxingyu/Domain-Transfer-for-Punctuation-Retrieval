#%%
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import BoolTensor, FloatTensor, LongTensor, ByteTensor
from typing import List, Optional, Sequence, Union, Callable, Dict, Any, Tuple
#%%
class DiceLoss(nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Focal Dice Loss
    Args:
        alpha (float): a factor to push down the weight of easy examples
        eps (float): a constant added to both the nominator and the denominator for smoothing purposes
        gamma(int): the index of exponentiation of numerator and denominator.
    """
    def __init__(self,
                 eps: Optional[float] = 0.05,
                 square_denominator: Optional[bool] = False,
                 gamma: Optional[int] = 1,
                 reduction: Optional[str] = "mean",
                 alpha: float = 1.0,
                 num_classes: int = 10,
                 weight=1, #int or list
                 ) -> None:
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.square_denominator = square_denominator
        self.weight=weight

    def forward(self,
                pred: FloatTensor,
                target: LongTensor,
                mask: ByteTensor,
                ) -> torch.Tensor:
        pred_soft = torch.softmax(pred,-1) #(batch_size,seq_len,num_labels)->(batch_size,seq_len,num_labels), sum along num_labels to 1
        target_one_hot=F.one_hot(target,num_classes=self.num_classes) #(b_s, s_l) -> (b_s, s_l, n_l)
        if mask is not None:
            pred_soft = pred_soft * mask.unsqueeze(-1)
            target_one_hot = target_one_hot * mask.unsqueeze(-1)
        pred_soft=pred_soft.flatten(end_dim=-2) #(b_s, s_l, n_l)-> (b_s*s_l, n_l)
        target_one_hot=target_one_hot.flatten(end_dim=-2) #(b_s, s_l, n_l)-> (b_s*s_l, n_l)
        intersection = torch.sum(pred_soft * target_one_hot, 0) # (b_s*s_l,n_l)->(n_l)
        cardinality = torch.sum(pred_soft + target_one_hot, 0) if self.square_denominator else torch.sum(pred_soft*pred_soft + target_one_hot*target_one_hot, 0) # (b_s*s_l,n_l)->(n_l)
        torch.pow(1. - 2. * (intersection + self.eps) / (cardinality + self.eps),2) * torch.tensor(1) # (b_s,n_l)->(b_s,n_l)
        dice_score = torch.pow(1. - (2. * (intersection ) + self.eps) / (cardinality + self.eps),self.gamma) * torch.tensor(self.weight) # (n_l),(n_l)->(n_l)
        if self.reduction == "mean":
            return dice_score.mean()
        elif self.reduction == "sum":
            return dice_score.sum()
        elif self.reduction == "none" or self.reduction is None:
            return dice_score
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
    def __str__(self):
        return f"Dice Loss eps:{self.eps}"
#%%
diceloss=DiceLoss(reduction='none',eps=0.02,gamma=3) #mean
print(diceloss(torch.tensor([0.0,0.0,1.0,0.0,0,0,0,0,0,0]),torch.tensor([1]),torch.tensor([1])))
print(diceloss(torch.tensor([0.0,1.0,0.0,0.0,0,0,0,0,0,0]),torch.tensor([1]),torch.tensor([1])))
#%%
class ElectraCRF(pl.LightningModule):
  def __init__(self, encoder: nn.Module=None, ):

#%%
