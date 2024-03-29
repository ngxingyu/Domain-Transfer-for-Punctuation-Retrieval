import torch

__all__ = ['CrossEntropyLoss']


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    CrossEntropyLoss
    """

    def __init__(self, logits_ndim=2, weight=None, reduction='mean'):
        """
        Args:
            logits_ndim (int): number of dimensions (or rank) of the logits tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
        """
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
        super().__init__(weight=weight, reduction='none')
        self._logits_dim = logits_ndim
        self.reduction=reduction

    def forward(self, logits, labels, loss_mask=None, token_weight=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return super().forward(logits, torch.argmax(logits, dim=-1))

        loss = super().forward(logits_flatten, labels_flatten)

        if token_weight is not None:
            loss=loss*token_weight
        return loss.mean() if self.reduction == 'mean'\
            else loss.sum() if self.reduction == 'sum'\
            else loss