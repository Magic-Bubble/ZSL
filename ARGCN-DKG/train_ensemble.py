import argparse
import json
import random
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from utils import ensure_path, set_gpu, l2_loss
from models.gat_relation import GAT
from models.gcn import GCN

def save_checkpoint(name):
    torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj1, osp.join(save_path, name + '.isa.pred'))
    torch.save(pred_obj2, osp.join(save_path, name + '.wup.pred'))

def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])

class Loss(nn.Module):
    def __init__(self, gcn, gat):
        super(Loss, self).__init__()
        self.gcn = gcn
        self.gat = gat
        self.alpha = nn.Parameter(torch.FloatTensor(1))
    
    def forward(self, word_vectors, fc_vectors, tlist, ntrain):
        output_vectors = [self.gcn(word_vectors), self.gat(word_vectors)]
        loss = torch.sigmoid(self.alpha) * mask_l2_loss(output_vectors[0], fc_vectors, tlist[:n_train]) + (1 - torch.sigmoid(self.alpha)) * mask_l2_loss(output_vectors[1], fc_vectors, tlist[:n_train])
        return loss
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=300)
    parser.add_argument('--save-path', default='save/ensemble')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    fc_vectors = torch.tensor(fc_vectors).float().cuda()
    fc_vectors = F.normalize(fc_vectors)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    hidden_layers = 'd2048,d'

    # for isa
    graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']

    edges = [tuple(e) for e in edges]
    edges += [(v, u) for (u, v) in edges]
    edges += [(u, u) for u in range(n)]
    weights = [1.0] * len(edges)

    word_vectors = graph['vectors']
    word_vectors = torch.tensor(word_vectors).float().cuda()
    word_vectors = F.normalize(word_vectors)

    assert train_wnids == wnids[:len(train_wnids)]

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    gcn = GCN(n, edges, weights, word_vectors, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers)

    # for wup
    graph = json.load(open('materials/imagenet-relation-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']
    weights = [1.0] * len(edges)

    word_vectors = graph['vectors']
    word_vectors = torch.tensor(word_vectors).float().cuda()
    word_vectors = F.normalize(word_vectors)

    assert train_wnids == wnids[:len(train_wnids)]

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    gat = GAT(n, edges, weights, word_vectors, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, tlist)

    model = Loss(gcn, gat).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        loss = model(word_vectors, fc_vectors, tlist, n_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.cpu().data.item()
        val_loss = 0
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'.format(epoch, train_loss, val_loss))

        if (epoch % args.save_epoch == 0):
            model.eval()
            output_vectors1 = model.gcn(word_vectors)
            output_vectors2 = model.gat(word_vectors)
            if args.no_pred:
                pred_obj1 = None
                pred_obj2 = None
            else:
                pred_obj1 = {
                    'wnids': wnids,
                    'pred': output_vectors1
                }
                pred_obj2 = {
                    'wnids': wnids,
                    'pred': output_vectors2
                }
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj1 = None
        pred_obj2 = None

