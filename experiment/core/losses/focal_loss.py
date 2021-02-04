import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional
from torch.nn.modules.loss import _WeightedLoss

__all__ = ['FocalLoss']

class FocalLoss(_WeightedLoss):

    __constants__ = ['gamma', 'reduction']
    gamma: int

    def __init__(self, weight: Optional[Tensor] = None, gamma=2, size_average=None, 
                ignore_index: int = -100, reduce=None, 
                 reduction: str = 'mean') -> None:         
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.gamma = gamma
    @snoop
    def forward(self, logits: Tensor, labels: Tensor, loss_mask=None) -> Tensor:

        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            logits_flatten=logits
            labels_flatten = torch.argmax(logits, dim=-1)

        logits_flatten_soft =F.log_softmax(logits_flatten,-1)
        target_one_hot=F.one_hot(labels_flatten,num_classes=logits_flatten_soft.shape[-1])
        weight = torch.tensor([1.]*self.num_classes).type_as(logits) if self.weight is None else self.weight

        # p_t = torch.where(target_one_hot==1, logits_flatten_soft, 1-logits_flatten_soft)
        p_t = logits_flatten_soft*target_one_hot
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(target == 1, fl * self.alpha, fl)

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return self._reduce(focal_loss)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x