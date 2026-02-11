from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import kahip

import parse_mnist
import parse_sift
import nlsh_build_and_search
from types import SimpleNamespace


def main(arguments):
    # Ensure every stochastic component (NumPy, PyTorch, KaHIP) uses the same seed per run.
    set_random_seeds(arguments.seed)

    # Step 1: load the binary dataset selected by the CLI into a float32 matrix.
    print("Loading dataset...")
    parse = parse_sift.parse if arguments.is_sift else parse_mnist.parse
    images = parse(arguments.input_dataset_filename)
    print(f"Dataset loaded. Shape: {images.shape}")

    # Step 2: build the weighted, symmetrised k-NN graph and partition it with KaHIP.
    # draw lines between every point and its k closest neighbors, then query KaHIP to
    # cut that graph into balanced groups (our "blocks").
    print("Building k-NN graph...")
    # Use n_jobs=-1 to use all CPU cores for faster k-NN search
    neighbors = get_neighbors(images, arguments.nearest_neighbors_for_knn_graph)
    print("Running KaHIP partitioning...")
    edgecut, blocks = use_kahip(arguments, neighbors)
    print("KaHIP finished.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The classifier predicts the KaHIP block id, so the output dimension equals the number of partitions.
    # Conceptually, the MLP is a student who looks at an image and tries to guess which group KaHIP (the teacher)
    # assigned it to. If the student can imitate the teacher well, we can later use its guesses to jump straight to
    # promising groups when answering queries.
    num_classes = int(blocks.max() + 1)
    assert num_classes == arguments.kahip_blocks
    model = nlsh_build_and_search.MLPClassifier(
        input_dim=images.shape[1],
        output_dim=num_classes,
        hidden_dim=arguments.nodes_per_layer,
        hidden_layers=arguments.mlp_layers,
    ).to(device)

    # Step 3: supervise the MLP with the KaHIP labels so it learns to predict partitions.
    # This is where the student repeatedly studies the dataset and gets graded against KaHIP's labels.
    loader, optimizer, loss_fn = init_training(images, blocks, model, arguments)

    train(model, loader, optimizer, loss_fn, device, arguments.epochs)

    inverted_index = build_inverted_index(blocks)
    # Step 4: save everything the search phase will need (student weights, teacher's labels, and index per block).
    state = {
        'model_state_dict': model.state_dict(),
        'partitions': blocks.tolist(),
        'inverted_index': inverted_index,
        'metadata': {
            'feature_dim': images.shape[1],
            'num_points': images.shape[0],
            'kahip': {
                'blocks': arguments.kahip_blocks,
                'imbalance': arguments.kahip_imbalance,
                'mode': arguments.kahip_mode,
                'edgecut': edgecut,
                'knn_graph_k': arguments.nearest_neighbors_for_knn_graph,
            },
            'mlp': {
                'layers': arguments.mlp_layers,
                'hidden_dim': arguments.nodes_per_layer,
                'epochs': arguments.epochs,
                'batch_size': arguments.batch_size,
                'learning_rate': arguments.learning_rate,
                'seed': arguments.seed,
            },
            'dataset': {
                'is_sift': arguments.is_sift,
                'path': arguments.input_dataset_filename,
            },
        },
    }
    torch.save(state, arguments.output_index_filename)


def get_neighbors(images, k):
    num_points = images.shape[0]
    if num_points == 0:
        raise ValueError('Dataset is empty; cannot build k-NN graph')

    neighbor_count = min(k + 1, num_points)
    # Use scikit-learn to sidestep reimplementing a k-NN graph; include self (k+1) so we can drop it later.
    # Each point writing down a list of its top-k closest neighbors.
    # n_jobs=-1 uses all available processors
    _, neighbor_indexes = NearestNeighbors(n_neighbors=neighbor_count, n_jobs=-1).fit(images).kneighbors(images)

    # Track how many times each undirected edge appears so we can detect mutual neighbors.
    # When A lists B and B lists A, the edge weight becomes 2 to show a stronger bond between them.
    edge_weights: Dict[tuple[int, int], int] = defaultdict(int)
    for i, neighbors in enumerate(neighbor_indexes):
        for neighbor in neighbors[1:]:
            j = int(neighbor)
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            edge_weights[edge] += 1

    adjacency: List[List[tuple[int, int]]] = [[] for _ in range(num_points)]
    for (u, v), count in edge_weights.items():
        # Mutual edges (observed twice) receive weight 2 per the assignment slides; unilateral edges remain 1.
        weight = 2 if count > 1 else 1
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    # Convert adjacency lists to CSR arrays expected by KaHIP.
    vwgt = [1] * num_points
    xadj = [0]
    adjncy: List[int] = []
    adjcwgt: List[int] = []
    for neighbors in adjacency:
        neighbors.sort()
        for vertex, weight in neighbors:
            adjncy.append(vertex)
            adjcwgt.append(weight)
        xadj.append(len(adjncy))

    return SimpleNamespace(vwgt=vwgt, xadj=xadj, adjncy=adjncy, adjcwgt=adjcwgt)


def use_kahip(arguments, r):

    suppress_output = True

    edgecut, one_block_label_per_point = kahip.kaffpa(
        r.vwgt,
        r.xadj,
        r.adjcwgt,
        r.adjncy,
        arguments.kahip_blocks,
        arguments.kahip_imbalance,
        suppress_output,
        arguments.seed,
        arguments.kahip_mode,
    )

    blocks = np.asarray(one_block_label_per_point, dtype=np.int64)
    return edgecut, blocks


def init_training(images, blocks, model, arguments):
    # Prepare tensors once so the DataLoader can stream batches without extra copies or dtype surprises.
    # building "flashcards" (features + labels) that the student MLP will study in mini-batches.
    features = torch.from_numpy(images.astype(np.float32))
    labels = torch.from_numpy(blocks.astype(np.int64))
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    return loader, optimizer, loss_fn


def train(model, loader, optimizer, loss_fn, device, epochs):
    # loop: iterate over epochs and backpropagate cross-entropy batches.
    # Each epoch lets the student review the entire flashcard set once; multiple epochs = multiple study sessions.
    model.train()  # set to training mode (because a model can also be in evaluation mode)
    for _ in range(epochs):
        for batch in loader:
            train_on_batch(model, batch, optimizer, loss_fn, device)


def train_on_batch(model, batch, optimizer, loss_fn, device):
    image, correct_label = batch
    image = image.to(device)
    correct_label = correct_label.to(device)

    # Standard cross-entropy training step.
    # This is one "pop quiz": compute guesses, compare to the teacher's answer, and nudge the student accordingly.
    optimizer.zero_grad()
    logits = model(image)
    loss = loss_fn(logits, correct_label)
    loss.backward()
    optimizer.step()


def build_inverted_index(labels):
    inverted: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        inverted[int(label)].append(idx)
    # Convert defaultdict to plain dict so it serialises cleanly in the checkpoint.
    return {int(label): ids for label, ids in inverted.items()}


def set_random_seeds(seed):
    # Keep NumPy, PyTorch, and CUDA RNGs aligned for reproducible partitions and training.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    from nlsh_build_args import parse_args
    main(parse_args())
