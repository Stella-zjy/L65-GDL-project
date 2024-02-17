import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, convert
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

from torch_geometric.nn import GCNConv

from sklearn import tree, linear_model

import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid
import umap

# from torch_geometric.nn import GNNExplainer

from utilities import *
from utils import pad_features
from heuristics import *
from activation_classifier import *
import models
from models import *

import random
import copy

set_rc_params()

# ensure reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# general parameters
dataset_name = "mutag"

model_type = Mutag_GCN_DiffPool8
load_pretrained = True

# hyperparameters
k = 40

# other parameters
train_test_split = 0.8
num_classes = 2

batch_size = 20

# small_loader is for visualization, making sure there are more than 100 graphs
small_loader_batch_size = 150

epochs = 1
lr = 0.005

# this is the trained model_name to load
model_name = "mutag_gcn_diffpool8_4_2_1_200epoch.pkl"

#  this is the activation_space for visualization
activation_space = 'conv3'

# in each cluster, view the subgraph of num_nodes_view number of nodes
num_nodes_view = 5
# n-hop subgraph
num_expansions = 4

paths = prepare_output_paths(dataset_name, k)

graphs = load_real_data("Mutagenicity")
train_loader, test_loader, full_loader, small_loader = prepare_real_data(graphs, 0.8, batch_size, "Mutagenicity",
                                                                         small_loader_batch_size)
small_loader_data = next(iter(small_loader)).x
small_loader_label = next(iter(small_loader)).y

labels = next(iter(full_loader)).y

model = model_type(graphs.num_node_features, graphs.num_classes)

if load_pretrained:

    print("Loading pretrained model...")
    model.load_state_dict(torch.load(os.path.join(paths['base'], model_name)))
    model.eval()


else:
    # model.apply(weights_init)
    train_graph_class(model, train_loader, test_loader, full_loader, small_loader, epochs, lr, paths['base'],
                      model_name)

print(test_graph_class(model, test_loader))
print(test_graph_class(model, train_loader))

# want to get class labels for nodes
full_class_labels = []
full_node_labels = []
full_class_labels_per_node = []

small_class_labels = []
small_node_labels = []
small_class_labels_per_node = []

small_node_labels_origin = []
small_class_labels_per_node_origin = []

# store the activation layers hen passing small_loader
model = register_hooks(model)
data = next(iter(small_loader))

# node_info have the information about which node is padded
batched_x, node_info = pad_features(data)

adj = to_dense_adj(data.edge_index, data.batch)
_ = model(batched_x, data.edge_index, data.batch, adj)
small_class_labels = data.y.detach().numpy()

for x, batch_idx in zip(batched_x, data.batch):
    small_node_labels.append(np.argmax(x.detach().numpy(), axis=0))
    small_class_labels_per_node.append(data.y[batch_idx])

# this is the original version, not batched, used for ploting activation space of the final conv layer
for x, batch_idx in zip(data.x, data.batch):
    small_node_labels_origin.append(np.argmax(x.detach().numpy(), axis=0))
    small_class_labels_per_node_origin.append(data.y[batch_idx])

# get the activation_space 
activation_list = models.activation_list

# check the keys of the stored activation_list
print("Available keys in activation_list:", activation_list.keys())
for key in activation_list:
    print(key, " ", activation_list[key].shape)

last_value = activation_list['conv3']
activation_list = {'conv3': last_value}

# TSNE conversion
tsne_models = []
tsne_data = []

for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    # get rid of the padded nodes
    recovered_activations = reshape_array_based_on_mask(activation, node_info)
    print(recovered_activations.shape)

    tsne_model = TSNE(n_components=2)

    d = tsne_model.fit_transform(recovered_activations)

    plot_activation_space(d, small_class_labels_per_node_origin, "t-SNE reduced", layer_num, paths['TSNE'],
                          "(coloured by class labels)", "_classlabels")

    tsne_models.append(tsne_model)
    tsne_data.append(d)

# PCA conversion
pca_models = []
pca_data = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    recovered_activations = reshape_array_based_on_mask(activation, node_info)
    pca_model = PCA(n_components=2)
    # n_samples, n_nodes, n_features = activation.shape
    # flattened_activation = activation.reshape(n_samples , n_nodes* n_features)
    # d = pca_model.fit_transform(flattened_activation)
    d = pca_model.fit_transform(recovered_activations)

    plot_activation_space(d, small_class_labels_per_node_origin, "PCA reduced", layer_num, paths['PCA'],
                          "(coloured by class labels)", "_classlabels")

    pca_models.append(pca_model)
    pca_data.append(d)

# UMAP conversion
umap_models = []
umap_data = []
for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    recovered_activations = reshape_array_based_on_mask(activation, node_info)
    umap_model = umap.UMAP(n_components=2)
    d = umap_model.fit_transform(recovered_activations)

    plot_activation_space(d, small_class_labels_per_node_origin, "UMAP reduced", layer_num, paths['UMAP'],
                          "(coloured by class labels)", "_classlabels")

    umap_models.append(umap_model)
    umap_data.append(d)

small_data = next(iter(small_loader))
small_edges = small_data.edge_index.transpose(0, 1).detach().numpy()

raw_kmeans_sample_feat = []
raw_kmeans_sample_graphs = []
raw_kmeans_models = []

for layer_num, key in enumerate(activation_list):
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation = reshape_array_based_on_mask(activation, node_info)
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(activation)
    pred_labels = kmeans_model.predict(activation)

    if key.find('conv') != -1:
        sample_labels = small_class_labels_per_node_origin
        sample_edges = small_edges
        node_labels = small_node_labels_origin

    #     plot_clusters(tsne_data[layer_num], pred_labels, "k-Means", k, layer_num, paths['KMeans'], "Raw", "_TSNE", "(t-SNE reduced)")
    #     plot_clusters(pca_data[layer_num], pred_labels, "k-Means", k, layer_num, paths['KMeans'], "Raw", "_PCA", "(PCA reduced)")
    #     plot_clusters(umap_data[layer_num], pred_labels, "k-Means", k, layer_num, paths['KMeans'], "Raw", "_UMAP", "(UMAP reduced)")
    sample_graphs, sample_feat = plot_samples(kmeans_model, activation, sample_labels, layer_num, k, "Kmeans", "raw",
                                              num_nodes_view, sample_edges, num_expansions, paths['KMeans'],
                                              node_labels, "Mutagenicity")

    raw_kmeans_sample_graphs.append(sample_graphs)
    raw_kmeans_sample_feat.append(sample_feat)
    raw_kmeans_models.append(kmeans_model)

