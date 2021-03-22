import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import transformer_weights_init

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=True):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.register_parameter('att_weights',nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True))
        nn.init.xavier_uniform_(self.att_weights.data)

        # self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def get_mask(self):
        pass

    def forward(self, hidden_states, attention_mask=None):

        if self.batch_first:
            batch_size, max_len = hidden_states.size()[:2]
        else:
            max_len, batch_size = hidden_states.size()[:2]

        # apply attention layer
        weights = torch.bmm(hidden_states,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )
        attentions = F.softmax(torch.tanh(weights.squeeze()),dim=-1)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * attention_mask
        if len(attentions.shape)==1:
            attentions=attentions.unsqueeze(0)
        _sums = masked.sum(-1,keepdim=True).expand(attentions.shape)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(hidden_states, attentions.unsqueeze(-1).expand_as(hidden_states))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze(dim=1)
        return representations, attentions