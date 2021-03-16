import torch
from torch import nn
from core.layers.multi_layer_perceptron import MultiLayerPerceptron
from core.layers.attention import SelfAttention
from core.utils import transformer_weights_init
# from nemo.core.neural_types import LabelsType, LogitsType, LossType, MaskType, NeuralType, LogprobsType
from typing import Optional, Dict

class SequenceClassifier(nn.Module):
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     if not self.log_softmax:
    #         return {"logits": NeuralType(('B', 'D'), LogitsType())}
    #     else:
    #         return {"log_probs": NeuralType(('B', 'D'), LogprobsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
        pooling: str = 'mean', # mean, max, mean_max, token
        idx_conditioned_on: int = None,
    ):
        """
        Initializes the SequenceClassifier module.
        Args:
            hidden_size: the hidden size of the mlp head on the top of the encoder
            num_classes: number of the classes to predict
            num_layers: number of the linear layers of the mlp head on the top of the encoder
            activation: type of activations between layers of the mlp head
            log_softmax: applies the log softmax on the output
            dropout: the dropout used for the mlp head
            use_transformer_init: initializes the weights with the same approach used in Transformer
            idx_conditioned_on: index of the token to use as the sequence representation for the classification task, default is the first token
        """
        super().__init__()
        self.log_softmax = log_softmax
        self._idx_conditioned_on = idx_conditioned_on
        self.pooling = pooling
        self.mlp = MultiLayerPerceptron(
            hidden_size=(hidden_size*2 if pooling=='mean_max' else hidden_size),
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        self.dropout=nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))
        if pooling=='attention':
            self.attention=SelfAttention(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.dropout(hidden_states)
        if self.pooling=='token':
            pooled = hidden_states[:, self._idx_conditioned_on]
        elif self.pooling=='attention':
            pooled, att = self.attention(hidden_states, attention_mask)
        else:
            if attention_mask is None:
                ct=hidden_states.shape[1] # Seq len
            else:
                hidden_states=hidden_states*attention_mask.unsqueeze(2) # remove subtoken or padding contribution.
                ct = torch.sum(attention_mask,axis=1).unsqueeze(1)
            pooled_sum = torch.sum(hidden_states,axis=1)

            if self.pooling=='mean' or self.pooling == 'mean_max':
                pooled_mean = torch.div(pooled_sum,ct)
            if self.pooling=='max' or self.pooling=='mean_max':
                pooled_max = torch.max(hidden_states,axis=1)[0]
            pooled=pooled_mean if self.pooling=='mean' else \
                pooled_max if self.pooling=='max' else \
                    torch.cat([pooled_mean,pooled_max],axis=-1)
        logits = self.mlp(pooled)
        return logits
