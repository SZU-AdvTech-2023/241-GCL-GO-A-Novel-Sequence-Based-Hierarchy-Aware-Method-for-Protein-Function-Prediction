import torch
import numpy as np
import scipy.sparse as sp

def one_hot_encode(go_id):
    # One-hot encoding of node features
    labels_onehot = np.identity(len(go_id), dtype=np.float)
    return torch.tensor(labels_onehot, dtype=torch.float)

def build_adj(idx, edge, go_id):
    # Building adjacency matrix
    adj_matrix = np.zeros((len(go_id), len(go_id)))
    for i, j in idx:
        adj_matrix[i][j] = edge[i]
        adj_matrix[j][i] = edge[i]
    return adj_matrix

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(axis=1))
    r_inv = np.power(rowsum, -1, where=rowsum!=0).flatten()
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))

def load_data(graph_dir, go_id, device):
    # Loading and preprocessing graph data
    idx, edge = [], []
    with open(graph_dir, "r") as file:
        for line in file:
            id1, id2, weight = line.strip().split('\t')
            idx.append([go_id.index(id1), go_id.index(id2)])
            edge.append(float(weight))
    
    adj = build_adj(idx, edge, go_id)
    adj = sp.coo_matrix(adj)
    adj_normalized = normalize(adj + adj.T.multiply(adj.T > adj) + sp.eye(adj.shape[0]))
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized).float().to(device)
    node_features = one_hot_encode(go_id)

    return adj_tensor, node_features

def make_neighbor_graph(h_semantic, go_id, sp_list, device):
    # Generating neighbor graph
    h_semantic_p = [torch.mean(torch.stack([h_semantic[go_id.index(sp_list[x][i])] for i in range(5)]), dim=0) for x in go_id]
    return torch.stack(h_semantic_p).to(device)
