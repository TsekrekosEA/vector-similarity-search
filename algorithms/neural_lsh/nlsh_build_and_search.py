from __future__ import annotations
from typing import List
import torch


# Multi-Layer Perceptron Classifier
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super().__init__()

        assert hidden_layers >= 1
        layers: List[torch.nn.Module] = []
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
