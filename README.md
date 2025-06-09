# Graph Convolutional Networks (GCN) Implementation

This repository contains a Jupyter notebook (`GCNs.ipynb`) implementing Graph Convolutional Networks (GCNs) for semi-supervised node classification, as described in the paper "Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf and Max Welling.

## Overview
The notebook implements a GCN model from scratch using PyTorch, with PyTorch Geometric used only for loading the Cora dataset. The model performs node classification by leveraging the graph structure and node features, using a small set of labeled nodes for training in a semi-supervised setting.

## Key Components

* **GCN Layer**: Implements the graph convolution operation with the normalized adjacency matrix and learnable weights, following the paper's update rule.
* **Model Architecture**: A two-layer GCN with ReLU activation, dropout for regularization, and a final classification layer.
* **Training**: Uses the Adam optimizer and cross-entropy loss, with performance evaluated on training and test sets.
* **Visualization**: Plots training loss and accuracy over epochs.

## Dataset
The Cora dataset is used, with:

* 2,708 nodes (scientific publications)
* 5,429 edges (citation links)
* 1,433 node features (bag-of-words)
* 7 classes (research topics)

The dataset is split into training, validation, and test sets, with only a portion of the nodes having labels for training.

## Implementation Details

### GCN Layer: 
Implements the update rule:
$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)
$$
where $\tilde{A} = A + I$ is the adjacency matrix with self-loops, $\tilde{D}$ is the degree matrix, $H^{(l)}$ is the feature matrix, $W^{(l)}$ is a learnable weight matrix, and $\sigma$ is ReLU for hidden layers.

### Model Hyperparameters:

* Number of layers: 2
* Hidden units: 16
* Dropout rate: 0.5
* Learning rate: 0.01
* Weight decay: 5e-4
* Epochs: 200


### Data Processing:

Loads the Cora dataset using `torch_geometric.datasets.Planetoid`.
Constructs the adjacency matrix from the edge index in COO format.

### Results
The model achieves a test accuracy of approximately 79.3% on the Cora dataset, aligning with the performance reported in the original paper for a two-layer GCN.

### References

* Kipf, T. N., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks." arXiv preprint arXiv:1609.02907.