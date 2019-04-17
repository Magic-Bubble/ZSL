import argparse
import json
import random
import os.path as osp

import torch
import torch.nn.functional as F

from utils import ensure_path, set_gpu, l2_loss
from models.gat import GAT


def save_checkpoint(name):
    torch.save(gat.state_dict(), osp.join(save_path, name + '.pth'))
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
    parser.add_argument('--save-path', default='save/gat-basic')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    ensure_path(save_path)

    graph = json.load(open('materials/imagenet-induced-graph.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']

    edges = [tuple(e) for e in edges]
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]
    weights = [1.0] * len(edges)

    word_vectors = torch.tensor(graph['vectors']).cuda()
    word_vectors = F.normalize(word_vectors)

    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    train_wnids = [x[0] for x in fcfile]
    fc_vectors = [x[1] for x in fcfile]
    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    hidden_layers = 'd2048,d'
    gat = GAT(n, edges, weights, word_vectors, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, tlist).cuda()

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gat.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gat.train()
        output_vectors = gat(word_vectors, train=True)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gat.eval()
        # output_vectors = gat(word_vectors)
        # train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        # if v_val > 0:
        #     val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
        #     loss = val_loss
        # else:
        #     val_loss = 0
        #     loss = train_loss
        train_loss = loss.item()
        val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'
              .format(epoch, train_loss, val_loss))

        # trlog['train_loss'].append(train_loss)
        # trlog['val_loss'].append(val_loss)
        # trlog['min_loss'] = min_loss
        # torch.save(trlog, osp.join(save_path, 'trlog'))

        if epoch % args.save_epoch == 0:
            gat.eval()
            output_vectors = gat(word_vectors)

            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None

