"""MLP classifier used by Neural LSH for learned space partitioning.

Algorithm Overview:
In classical LSH the hash function is a random projection, as described in lsh.hpp.
Neural LSH replaces that with a learned hash: a multi-layer perceptron that maps a
feature vector to one of M partitions discovered by balanced graph partitioning via
KaHIP.  At search time the MLP predicts the T most-likely partitions for a query,
and only the points assigned to those partitions are scanned, analogous to the
nprobe parameter in IVF-Flat (see ivfflat.hpp).

Theoretical Foundation:
Graph partitioning with KaHIP produces a balanced partition that minimises the edge
cut on the mutual k-NN graph.  Points that are near each other in Euclidean space
tend to appear in the same partition.  The MLP is trained via cross-entropy on these
partition labels, which amounts to knowledge distillation: the combinatorial
partitioner acts as the teacher and transfers its structure to the lightweight neural
network, the student.  At inference the softmax output distribution directly encodes
the probability of each partition, and the top-T predictions serve as multi-probe
indices.

Architecture Rationale:
A simple stack of (Linear, ReLU) blocks followed by a final Linear layer is chosen
because the MLP inference for a batch of queries amounts to a few matrix multiplies,
far cheaper than re-running KaHIP.  The hidden_dim and hidden_layers parameters let
us trade model size for partition-prediction accuracy, and deterministic weights under
torch.manual_seed ensure reproducibility.  For MNIST at 784 dimensions, hidden_dim=64
with 3 hidden layers yields roughly 3K parameters.  For SIFT at 128 dimensions,
hidden_dim=128 with 3 hidden layers yields roughly 50K parameters.

Complexity Analysis:
Forward pass runs in O(D*H + (H-1)*H^2 + H*M) where D is the input dimensionality,
H is the hidden dimension, and M is the number of output partitions.  Memory for
the weight matrices is O(D*H + H^2 + H*M) in float32.
"""

from __future__ import annotations

from typing import List

import torch


class MLPClassifier(torch.nn.Module):
    """Multi-layer perceptron that maps a feature vector to a KaHIP partition id.

    The network is a simple stack of (Linear, ReLU) blocks followed by a final
    linear projection to output_dim logits.  The logits are consumed by torch.topk
    at search time, where no softmax is needed since topk only requires monotonic
    values, or by CrossEntropyLoss during training, which internally applies
    log-softmax.  The input_dim parameter specifies the dimensionality of feature
    vectors, for example 784 for MNIST.  The output_dim determines the number of
    KaHIP partitions.  The hidden_dim controls the width of every hidden layer,
    where wider layers increase capacity but also inference cost; 64 to 128 is
    typical for the datasets in this project.  The hidden_layers parameter sets the
    number of (Linear, ReLU) blocks and must be at least 1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        assert hidden_layers >= 1, "Need at least one hidden layer"

        layers: List[torch.nn.Module] = []
        in_features = input_dim
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(in_features, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_features = hidden_dim
        layers.append(torch.nn.Linear(in_features, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, output_dim).

        No activation is applied after the last linear layer because CrossEntropyLoss
        expects unnormalised scores, and torch.topk only needs monotonic values.
        """
        return self.net(x)
