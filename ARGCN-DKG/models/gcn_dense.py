import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor

# class SpecialSpmmFunction(torch.autograd.Function):
# 
#     """Special function for only sparse region backpropataion layer."""
# 
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)
# 
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b
# 
# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)
# 
# class SpGraphAttentionLayer(nn.Module):
# 
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
# 
#     def __init__(self, in_channels, out_channels, dropout=False, relu=True):
#         super().__init__()
# 
#         if dropout:
#             self.dropout = nn.Dropout(p=0.5)
#         else:
#             self.dropout = None
# 
#         self.qw = nn.Parameter(torch.empty(in_channels, out_channels))
#         xavier_uniform_(self.qw)
# 
#         self.kw = nn.Parameter(torch.empty(in_channels, out_channels))
#         xavier_uniform_(self.kw)
# 
#         self.vw = nn.Parameter(torch.empty(in_channels, out_channels))
#         xavier_uniform_(self.vw)
# 
#         self.a = nn.Parameter(torch.empty(1, 2 * out_channels))
#         xavier_uniform_(self.a)
# 
#         if relu:
#             self.relu = nn.LeakyReLU(negative_slope=0.2)
#         else:
#             self.relu = None
# 
#         self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
# 
#         self.special_spmm = SpecialSpmm()
# 
#     def forward(self, inputs, adj):
#         N = inputs.size()[0]
# 
#         if self.dropout is not None:
#             inputs = self.dropout(inputs)
# 
#         edge = adj._indices()
# 
#         qh = torch.mm(inputs, self.qw)
#         assert not torch.isnan(qh).any()
# 
#         kh = torch.mm(inputs, self.kw)
#         assert not torch.isnan(kh).any()
# 
#         vh = torch.mm(inputs, self.vw)
#         assert not torch.isnan(vh).any()
# 
#         # Self-attention on the nodes - Shared attention mechanism
#         edge_h = torch.cat((qh[edge[0, :], :], kh[edge[1, :], :]), dim=1).t()
#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
# 
#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
#         mask = torch.ones(e_rowsum.size()).cuda()
#         mask[torch.nonzero(e_rowsum)] = 0.0
#         e_rowsum += mask
# 
#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), vh)
#         assert not torch.isnan(h_prime).any()
# 
#         h_prime = h_prime.div(e_rowsum)
#         assert not torch.isnan(h_prime).any()
# 
#         if self.relu is not None:
#             h_prime = self.relu(h_prime)
# 
#         return h_prime


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense(nn.Module):

    def __init__(self, n, edges, weights, in_channels, out_channels, hidden_layers, train_idx):
        super().__init__()

        edges = np.array(edges)

        # neighs = {}
        # for edge in edges:
        #     if edge[0] not in neighs:
        #         neighs[edge[0]] = []
        #     neighs[edge[0]].append(edge[1])
        # neighs = {k: set(neighs[k]) for k in neighs}

        # weights = np.array(weights)
        weights = np.ones(len(weights))
        self.adj = self.get_adj(weights, edges, n)
        print(self.adj)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

        # get simple graph for train only
        # self.train_adj = []

        # train_idx = set(train_idx)
        # for ii in range(len(layers)):
        #     new_neighs = set()
        #     for idx in train_idx:
        #         new_neighs |= neighs[idx]
        #     train_idx |= new_neighs

        #     train_edges = []
        #     train_weights = []
        #     for i, edge in enumerate(edges):
        #         if edge[0] in train_idx and edge[1] in train_idx:
        #             train_edges.append(edge)
        #             train_weights.append(weights[i])
        #     train_edges = np.array(train_edges)
        #     train_weights = np.array(train_weights)
        #     print('layer {}, num edges {}'.format(len(layers)-ii, len(train_edges)))
        #     self.train_adj.insert(0, self.get_adj(train_weights, train_edges, n))

    def get_adj(self, weights, edges, n):
        adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        return adj.cuda()

    def forward(self, x, train=False):
        # if train:
        #     for i, conv in enumerate(self.layers):
        #         x = conv(x, self.train_adj[i])
        # else:
        #     with torch.no_grad():
        #         for conv in self.layers:
        #             x = conv(x, self.adj)

        for conv in self.layers:
            x = conv(x, self.adj)

        return F.normalize(x)

