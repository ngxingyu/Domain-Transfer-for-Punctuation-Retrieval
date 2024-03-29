import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional

__all__ = ['FocalLoss']

class FocalLoss(torch.nn.CrossEntropyLoss):

    __constants__ = ['alpha', 'reduction']
    alpha: int

    def __init__(self, weight = None, alpha=2, size_average=None, 
                ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:         
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
        super(FocalLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits: Tensor, labels: Tensor, loss_mask=None, token_weight=None) -> Tensor:
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2) #try change from -2 to 1
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1) #try change from -1 to 1
        self.num_classes=logits_flatten.shape[-1]
        
        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            logits_flatten=logits
            labels_flatten = torch.argmax(logits, dim=-1)

        logits_flatten_soft = F.log_softmax(logits_flatten,-1)
        ce = super().forward(logits_flatten_soft,labels_flatten)
        all_rows = torch.arange(len(logits_flatten))
        log_pt = logits_flatten_soft[all_rows, labels_flatten]

        # compute focal term: (1 - pt)^alpha
        pt = log_pt.exp()
        focal_term = torch.pow(1 - pt,self.alpha)

        loss = focal_term * ce

        if token_weight is not None:
            loss=loss*token_weight
        return loss.mean() if self.reduction == 'mean'\
            else loss.sum() if self.reduction == 'sum'\
            else loss