import argparse
import json
import os.path as osp
import torch
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
from collections import Counter
from nltk.corpus import wordnet as wn

from utils import pick_vectors, set_gpu, load_X

def get_node(wnid):
    return wn.synset_from_pos_and_offset('n', int(wnid[1:]))

def get_wnids():
    suppl = h5py.File('/home/t-yud/ZSL/data/ImageNet/ImageNet_w2v/ImageNet_w2v.mat')
    wnids = [''.join(chr(i) for i in suppl[wnid[0]]) for wnid in suppl['wnids']]
    print('wnids:', len(wnids), wnids[:10])
    return wnids

def get_subset(label):
    feature = load_X('/home/t-yud/ZSL/data/ImageNet/res101_1crop_feature/{}.bin'.format(label))
    feature = torch.from_numpy(feature).float().cuda()
    return feature

def test_on_subset(feat, n, pred_vectors, all_label, alpha,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0

    fcs = [pred_vector.t() for pred_vector in pred_vectors]
    table = [torch.matmul(feat, fc) for fc in fcs]
    table = alpha * table[0] + (1 - alpha) * table[1]
    # table = torch.stack(table, dim=2).mean(2)

    if not consider_trains:
        table[:, :n] = -1e18

    gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
    rks = (table >= gth_score).sum(dim=1)

    assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

    for i, k in enumerate(top):
        hits[i] += (rks <= k).sum().item()
    tot += len(feat)

    return hits, tot

def test_badcase(feat, n, pred_vectors, nodes):
    fcs = pred_vectors.t()
    table = torch.matmul(feat, fcs)
    print('Total test sample:', len(feat))
    
    table[:, :n] = -1e18

    top = table.max(1)[1]
    top_node = [nodes[t] for t in top]

    top_sta = Counter(top_node)
    top_sta = list(sorted(top_sta.items(), key=lambda x: x[1], reverse=True))
    print(top_sta[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', nargs='+')
    parser.add_argument('--test-set')
    parser.add_argument('--output', default=None)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--test-train', action='store_true')
    parser.add_argument('--test-badcase', action='store_true')
    parser.add_argument('--test-wnid', default=None)

    args = parser.parse_args()

    set_gpu(args.gpu)

    test_sets = json.load(open('materials/imagenet-testsets.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets[args.test_set]

    print('test set: {}, {} classes'.format(args.test_set, len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_files = [torch.load(pred) for pred in args.pred]
    pred_wnids = [pred_file['wnids'] for pred_file in pred_files]
    pred_vectors = [pred_file['pred'] for pred_file in pred_files]
    pred_dics = [dict(zip(pred_wnids[ii], pred_vectors[ii])) for ii in range(len(pred_wnids))]
    pred_vectors = [pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda() for pred_dic in pred_dics]
    nodes = [get_node(x) for x in train_wnids + test_wnids]

    n = len(train_wnids)
    m = len(test_wnids)
    
    TEST_TRAIN = args.test_train

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda() # top 1 2 5 10 20
    s_tot = 0

    results = {}

    wnids = get_wnids()

    if args.test_badcase:
        node = get_node(args.test_wnid)
        print('test bad case on', node)
        subset = get_subset(wnids.index(args.test_wnid)+1)
        test_badcase(subset, n, pred_vectors, nodes)
        import sys; sys.exit(0)

    if TEST_TRAIN:
        for i, wnid in enumerate(train_wnids, 1):
            subset = get_subset(wnids.index(wnid)+1)
            hits, tot = test_on_subset(subset, n, pred_vectors, i - 1,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(train_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))
    else:
        for i, wnid in enumerate(test_wnids, 1):
            subset = get_subset(wnids.index(wnid)+1)
            hits, tot = test_on_subset(subset, n, pred_vectors, n + i - 1, alpha=args.alpha,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(test_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))

    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ')
    print('total {}'.format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))

