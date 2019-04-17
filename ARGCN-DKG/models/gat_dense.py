import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor

class SpecialSpmmFunction(torch.autograd.Function):

    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):

    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None


        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        xavier_uniform_(self.w)

        self.a = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.BatchNorm1d(out_channels//2),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels//2, out_channels//4),
            nn.BatchNorm1d(out_channels//4),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels//4, 1),
            nn.LeakyReLU(0.2)
        )

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

        self.special_spmm = SpecialSpmm()

    def forward(self, inputs, adj):
        N = inputs.size()[0]

        if self.dropout is not None:
            inputs = self.dropout(inputs)

        edge = adj._indices()

        h = torch.mm(inputs, self.w)
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.abs(h[edge[0, :], :] - h[edge[1, :], :])
        edge_e = self.a(edge_h).squeeze(1)
        edge_e = torch.exp(-edge_e)
        edge_e = edge_e + (edge[0] == edge[1]).float()
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        mask = torch.ones(e_rowsum.size()).cuda()
        mask[torch.nonzero(e_rowsum)] = 0.0
        e_rowsum += mask

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.relu is not None:
            h_prime = self.relu(h_prime)

        return h_prime

class GAT_Dense(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, train_idx):
        super().__init__()

        edges = np.array(edges)

        neighs = {}
        for edge in edges:
            if edge[0] not in neighs:
                neighs[edge[0]] = []
            neighs[edge[0]].append(edge[1])
        neighs = {k: set(neighs[k]) for k in neighs}

        self.adj, self.r_adj = self.get_adj(n, edges)

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
            conv = SpGraphAttentionLayer(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = SpGraphAttentionLayer(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

        # get simple graph for train only
        self.train_adj, self.train_r_adj = [], []
        train_idx = set(train_idx)
        for ii in range(len(layers)):
            new_neighs = set()
            for idx in train_idx:
                new_neighs |= neighs[idx]
            train_idx |= new_neighs

            train_edges = []
            for edge in edges:
                if edge[0] in train_idx and edge[1] in train_idx:
                    train_edges.append(edge)
            train_edges = np.array(train_edges)
            print('layer {}, num edges {}'.format(len(layers)-ii, len(train_edges)))
            adj, r_adj = self.get_adj(n, train_edges)
            self.train_adj.insert(0, adj)
            self.train_r_adj.insert(0, r_adj)

    def get_adj(self, n, edges):
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj_ = spm_to_tensor(normt_spm(adj, method='in')).cuda()
        r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in')).cuda()
        return adj_, r_adj

    def forward(self, x, train=True):
        if train:
            graph_side = True
            for i, conv in enumerate(self.layers):
                if graph_side:
                    x = conv(x, self.train_adj[i])
                else:
                    x = conv(x, self.train_r_adj[i])
                graph_side = not graph_side
        else:
            graph_side = True
            with torch.no_grad():
                for conv in self.layers:
                    if graph_side:
                        x = conv(x, self.adj)
                    else:
                        x = conv(x, self.r_adj)
                graph_side = not graph_side
        return F.normalize(x)

