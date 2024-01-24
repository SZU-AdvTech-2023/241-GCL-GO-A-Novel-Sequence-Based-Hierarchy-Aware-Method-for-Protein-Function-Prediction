import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adjacency):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adjacency, support)
        return output + self.bias if self.bias is not None else output

class SequenceEncoder(nn.Module):
    def __init__(self, nhid, dropout):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(1280, nhid[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid[0], nhid[1]*2)
        )

    def forward(self, sequence_embedding):
        return self.fc_layers(sequence_embedding)

class GraphNeuralNetwork(nn.Module):
    def __init__(self, seq_feature, go_feature, nhid, kernel_size, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(go_feature, nhid[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid[0], nhid[1])
        )
        self.graph_conv1 = GraphConvolution(go_feature, nhid[0])
        self.graph_conv2 = GraphConvolution(nhid[0], nhid[1])
        self.sequence_encoder = SequenceEncoder(nhid, dropout)

    def forward(self, sequence_embedding, go_embedding, adjacency_matrix):
        h_semantic = self.mlp(go_embedding)
        x = F.relu(self.graph_conv1(go_embedding, adjacency_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        h_structure = F.relu(self.graph_conv2(x, adjacency_matrix))

        seq_output = self.sequence_encoder(sequence_embedding)
        go_output = torch.cat([h_semantic, h_structure], dim=1).transpose(0, 1)

        prediction = torch.mm(seq_output, go_output)
        return h_semantic, h_structure, torch.sigmoid(prediction)
