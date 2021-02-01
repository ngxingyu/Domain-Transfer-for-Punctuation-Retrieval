import torch
from core.layers.multi_layer_perceptron import MultiLayerPerceptron
from core.utils import transformer_weights_init

class SequenceClassifier(torch.nn.Module):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if not self.log_softmax:
            return {"logits": NeuralType(('B', 'D'), LogitsType())}
        else:
            return {"log_probs": NeuralType(('B', 'D'), LogprobsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
        idx_conditioned_on: int = 0,
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
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.log_softmax = log_softmax
        self._idx_conditioned_on = idx_conditioned_on
        self.mlp = MultiLayerPerceptron(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states[:, self._idx_conditioned_on])
        return logits
