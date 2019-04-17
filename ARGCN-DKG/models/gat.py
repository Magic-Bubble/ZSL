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
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        # self.a = nn.Parameter(torch.empty(out_channels * 2, 1))
        self.a = nn.Parameter(torch.empty(in_channels, 1))
        xavier_uniform_(self.a)

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

        # h = torch.mm(inputs, self.w)
        # assert not torch.isnan(h).any()

        # edge_e = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1)
        # edge_e = torch.exp(-F.leaky_relu(torch.mm(edge_e, self.a), 0.2)).squeeze()
        edge_e = torch.mm(inputs[edge[0, :], :], self.a) + torch.mm(inputs[edge[1, :], :], self.a)
        edge_e = torch.exp(-F.leaky_relu(edge_e, 0.2)).squeeze()
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        mask = torch.ones(e_rowsum.size()).cuda()
        mask[torch.nonzero(e_rowsum)] = 0.0
        e_rowsum = e_rowsum + mask

        # h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        # assert not torch.isnan(h_prime).any()

        # h_prime = h_prime.div(e_rowsum)
        # assert not torch.isnan(h_prime).any()

        neighs = self.special_spmm(edge, edge_e, torch.Size([N, N]), inputs)
        assert not torch.isnan(neighs).any()

        neighs = neighs.div(e_rowsum)
        assert not torch.isnan(neighs).any()

        outputs = torch.mm(neighs, self.w) + torch.mm(inputs, self.w) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)

        return outputs


class GAT(nn.Module):

    def __init__(self, n, edges, weights, word_vectors, in_channels, out_channels, hidden_layers, train_idx):
        super().__init__()

        edges = np.array(edges)

        self.adj = self.get_adj(np.ones(len(edges)), edges, n)

        neighs = {}
        for edge in edges:
            if edge[0] not in neighs:
                neighs[edge[0]] = []
            neighs[edge[0]].append(edge[1])
        neighs = {k: set(neighs[k]) for k in neighs}

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
        self.train_adj = []

        train_idx = set(train_idx)
        for ii in range(len(layers)):
            new_neighs = set()
            for idx in train_idx:
                new_neighs |= neighs[idx]
            train_idx |= new_neighs

            train_edges = []
            for j, edge in enumerate(edges):
                if edge[0] in train_idx and edge[1] in train_idx:
                    train_edges.append(edge)
            train_edges = np.array(train_edges)
            print('layer {}, num edges {}'.format(len(layers)-ii, len(train_edges)))
            self.train_adj.insert(0, self.get_adj(np.ones(len(train_edges)), train_edges, n))

    def get_adj(self, weights, edges, n):
        adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        return adj.cuda()

    def forward(self, x, train=False):
        if train:
            for i, conv in enumerate(self.layers):
                x = conv(x, self.train_adj[i])
        else:
            with torch.no_grad():
                for conv in self.layers:
                    x = conv(x, self.adj)
        return F.normalize(x)
