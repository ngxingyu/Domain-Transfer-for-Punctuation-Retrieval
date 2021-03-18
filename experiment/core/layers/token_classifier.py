# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Modifications Copyright (c) 2021 <Ng Xing Yu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn
from core.layers.multi_layer_perceptron import MultiLayerPerceptron
from core.utils import transformer_weights_init
from typing import Optional, Dict

class TokenClassifier(nn.Module):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

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
        super().__init__()
        self.log_softmax = log_softmax
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=num_layers, activation=activation, log_softmax=log_softmax
        )
        self.dropout=nn.Dropout(dropout)
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