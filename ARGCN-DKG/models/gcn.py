import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True, residual=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.residual = residual

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
       
        if self.residual:

            neighs = torch.mm(adj, inputs)
            outputs = torch.mm(neighs, self.w) + torch.mm(inputs, self.w) + self.b

        else:
            
            outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN(nn.Module):

    def __init__(self, n, edges, weights, word_vectors, in_channels, out_channels, hidden_layers, residual=True):
        super().__init__()
       
        edges = np.array(edges)

        uniform_adj = self.get_adj(np.ones(len(edges)), edges, n)
        self.adj = uniform_adj
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
            conv = GraphConv(last_c, c, dropout=dropout, residual=residual)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def get_adj(self, weights, edges, n):
        adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        return adj.cuda()

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.adj)
        return F.normalize(x)
