import torch
from core.layers.multi_layer_perceptron import MultiLayerPerceptron
from core.utils import transformer_weights_init

class TokenClassifier(torch.nn.Module):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns definitions of module output ports.
        """
        if not self.log_softmax:
            return {"logits": NeuralType(('B', 'T', 'C'), LogitsType())}
        else:
            return {"log_probs": NeuralType(('B', 'T', 'C'), LogprobsType())}

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ) -> None:

        """
        Initializes the Token Classifier module.
        Args:
            hidden_size: the size of the hidden dimension
            num_classes: number of classes
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output of the MLP
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__(hidden_size=hidden_size, dropout=dropout)
        self.log_softmax = log_softmax
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def forward(self, hidden_states):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits