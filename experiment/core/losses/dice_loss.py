import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional
from torch.nn.modules.loss import _WeightedLoss

__all__ = ['FocalDiceLoss']

class FocalDiceLoss(_WeightedLoss):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        epsilon (float): a factor added to both the nominator and the denominator for smoothing purposes

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = (1-\frac{2\cdot x \cdot class + \epsilon}{x + \class +\epsilon})^{\alpha}

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = \weight[class] (1-\frac{2\cdot x \cdot class + \epsilon}{x + \class +\epsilon})^{\alpha}
        weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch. If the
    :attr:`weight` argument is specified then this is a weighted average:

    .. math::
        \text{loss} = \frac{\sum^{N}_{i=1} loss(i, class[i])}{\sum^{N}_{i=1} weight[class[i]]}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> loss = FocalDiceLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'macro_average', 'alpha', 'epsilon', 'square_denominator']
    ignore_index: int
    macro_average: bool
    alpha: int
    epsilon: float
    square_denominator: bool

    def __init__(self, weight: Optional[Tensor] = None, num_labels:int=10, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', macro_average=True, alpha = 1.0, beta = 1.0, epsilon = 0.05, 
                 square_denominator = False, log_softmax=False) -> None:
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
        super(FocalDiceLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.num_classes=num_labels
        self.macro_average = macro_average
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.square_denominator = square_denominator
        self.log_softmax = log_softmax

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

        logits_flatten_soft = F.log_softmax(logits_flatten,-1) if self.log_softmax else F.softmax(logits_flatten,-1) #(batch_size,seq_len,num_labels)->(batch_size,seq_len,num_labels), sum along num_labels to 1
        target_one_hot=F.one_hot(labels_flatten,num_classes=logits_flatten_soft.shape[-1]) #(b_s, s_l) -> (b_s, s_l, n_l)        
        TP = torch.sum(logits_flatten_soft * target_one_hot,0)    # (b_s*s_l,n_l)->(n_l)
        FP = torch.sum(logits_flatten_soft * (1-target_one_hot),0)    # (b_s*s_l,n_l)->(n_l)
        FN = torch.sum((1-logits_flatten_soft) * target_one_hot,0)    # (b_s*s_l,n_l)->(n_l)
        intersection = TP
        # cardinality = torch.sum(logits_flatten_soft*logits_flatten_soft + target_one_hot*target_one_hot, 0) if self.square_denominator else torch.sum(logits_flatten_soft + target_one_hot, 0) # (b_s*s_l,n_l)->(n_l)
        cardinality = (1.+self.beta**2)*TP + self.beta**2 * FN + FP # (b_s*s_l,n_l)->(n_l)
        weight = torch.tensor([1.]*self.num_classes).type_as(logits) if self.weight is None else self.weight

        if self.macro_average:
            focal_dice_score = torch.pow(1. - (1.+self.beta**2) * (intersection + self.epsilon) / (cardinality + self.epsilon),self.alpha) * weight # (n_l),(n_l)->(n_l)
        else:
            focal_dice_score = torch.pow(1. - (1.+self.beta**2) * (torch.dot(intersection,weight) + self.epsilon) / (torch.dot(cardinality,weight) + self.epsilon),self.alpha) # (n_l),(n_l)->(n_l)
        if self.reduction == "mean":
            return focal_dice_score.sum()/weight.sum()
        elif self.reduction == "sum":
            return focal_dice_score.sum()
        elif self.reduction == "none" or self.reduction is None:
            return focal_dice_score
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
