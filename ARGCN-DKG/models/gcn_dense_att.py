import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, d, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        self.a_att = nn.Parameter(torch.ones(d)) 

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        support = torch.mm(inputs, self.w) + self.b
        outputs = None

        a_att = F.softmax(self.a_att, dim=0)
        for i, adj in enumerate(adj_set):
            y = torch.mm(adj, support) * a_att[i]
            if outputs is None:
                outputs = y
            else:
                outputs += y

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense_Att(nn.Module):

    def __init__(self, n, edges_set, weights_set, in_channels, out_channels, hidden_layers):
        super().__init__()

        self.n = n
        self.d = len(edges_set)

        self.a_adj_set = []

        for i, edges in enumerate(edges_set):
            edges = np.array(edges)
            adj = sp.coo_matrix((np.array(weights_set[i]), (edges[:, 0], edges[:, 1])),
                                shape=(n, n), dtype='float32')
            a_adj = spm_to_tensor(normt_spm(adj, method='in')).cuda()
            self.a_adj_set.append(a_adj)

        print(self.a_adj_set)

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
            conv = GraphConv(last_c, c, self.d, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, self.d, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.a_adj_set)

        return F.normalize(x)

