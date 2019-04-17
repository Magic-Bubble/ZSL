import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gcn_dense_att import GCN_Dense_Att


def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(save_path, name + '.pth'))
    torch.save(pred_obj, osp.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-epoch', type=int, default=300)
    parser.add_argument('--save-path', default='save/gcn-dense-att')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    graph = json.load(open('materials/imagenet-dense-grouped-graph-weighted.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)

    edges_set = graph['edges_set']
    weights_set = graph['weights_set']
    print('edges_set', [len(l) for l in edges_set])

    # weight group 0.9-1.0 0.8-0.9 0.7-0.8 ... not well, attention focus exactlay on the previous one
    # weight group 0.99-1.0 0.98-0.99 ...
    edges_sim_group = []
    weights_sim_group = []
    for _ in range(11):
        edges_sim_group.append([])
        weights_sim_group.append([])
    for d in range(len(edges_set)):
        edges = edges_set[d]
        weights = weights_set[d]
        for ii in range(len(weights)):
            if weights[ii] < 0.9: continue
            block = int((1 - weights[ii]) * 100)
            weights_sim_group[block].append(weights[ii])
            edges_sim_group[block].append(edges[ii])
    edges_set = edges_sim_group
    weights_set = weights_sim_group

    lim = 4
    for i in range(lim + 1, len(edges_set)):
        edges_set[lim].extend(edges_set[i])
        weights_set[lim].extend(weights_set[i])
    edges_set = edges_set[:lim + 1]
    weights_set = weights_set[:lim + 1]
    # add self-loop and symmetic
    for dist in range(len(edges_set)):
        # if dist == 0: continue
        edges_set[dist] = [tuple(edge) for edge in edges_set[dist]]
        edges_set[dist] += [(v, u) for u, v in edges_set[dist]]
        # weights_set[dist] += weights_set[dist]
        edges_set[dist] += [(u, u) for u in range(n)]
        edges_set[dist] = list(set(edges_set[dist]))
        # weights_set[dist] += [1.0] * n
        weights_set[dist] = [1.0] * len(edges_set[dist])
    print('edges_set', [len(l) for l in edges_set])
    
    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    hidden_layers = 'd2048,d'
    gcn = GCN_Dense_Att(n, edges_set, weights_set,
                        word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()

    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gcn.eval()
        output_vectors = gcn(word_vectors)
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
              .format(epoch, train_loss, val_loss))

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog'))

        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }

        if epoch % args.save_epoch == 0:
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None

