import torch
from typing import List

__all__ = ['AggregatorLoss']


class AggregatorLoss(torch.nn.modules.loss._Loss):
    """
    Sums several losses into one.
    Args:
        num_inputs: number of input losses
        weights: a list of coefficient for merging losses
    """

    def __init__(self, num_inputs: int = 2, weights: List[float] = None):
        super(AggregatorLoss, self).__init__()
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise ValueError("Length of weights should be equal to the number of inputs (num_inputs)")

        self._weights = weights

    def forward(self, **kwargs):
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weights[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss