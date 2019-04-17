import argparse
import json
import os.path as osp
import numpy as np

import torch
from torch.nn import functional as F

from scipy.io import loadmat
from utils import set_gpu, pick_vectors

def load_feature():
    res101 = loadmat('../data/AWA2/res101.mat')
    att = loadmat('../data/AWA2/att_splits.mat')
    classes = [f.item() for f in att['allclasses_names'].reshape(-1)]
    labels = res101['labels'].reshape(-1)
    features = res101['features'].T
    feature_dict = {}
    for c in classes:
        feature_dict[c] = []
    for ii in range(len(labels)):
        feature_dict[classes[labels[ii]-1]].append(features[ii])
    for c in feature_dict:
        feature_dict[c] = np.array(feature_dict[c])
    print(len(feature_dict['sheep']))
    return feature_dict

def test_on_subset(feat, n, pred_vectors, all_label,
                   consider_trains):
    hit = 0
    tot = 0

    feat = feat.float().cuda()

    fcs = [pred_vector.t() for pred_vector in pred_vectors]
    table = [torch.matmul(feat, fc) for fc in fcs]
    table = torch.stack(table, dim=2).mean(2)
    if not consider_trains:
        table[:, :n] = -1e18

    pred = torch.argmax(table, dim=1)

    hit += (pred == all_label).sum().item()
    tot += len(feat)

    return hit, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', nargs='+')

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    set_gpu(args.gpu)

    awa2_split = json.load(open('materials/awa2-split.json', 'r'))
    train_wnids = awa2_split['train']
    test_wnids = awa2_split['test']

    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))

    
    pred_files = [torch.load(pred) for pred in args.pred]
    pred_wnids = [pred_file['wnids'] for pred_file in pred_files]
    pred_vectors = [pred_file['pred'] for pred_file in pred_files]
    pred_dics = [dict(zip(pred_wnids[ii], pred_vectors[ii])) for ii in range(len(pred_wnids))]
    pred_vectors = [pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda() for pred_dic in pred_dics]

    n = len(train_wnids)
    m = len(test_wnids)
    
    test_names = awa2_split['test_names']

    ave_acc = 0; ave_acc_n = 0

    results = {}

    feat_dict = load_feature()

    for i, name in enumerate(test_names, 1):
        print(feat_dict[name].shape)
        feat = torch.from_numpy(feat_dict[name])
        hit, tot = test_on_subset(feat, n, pred_vectors, n + i - 1,
                                  args.consider_trains)
        acc = hit / tot
        ave_acc += acc
        ave_acc_n += 1

        print('{} {}: {:.2f}%'.format(i, name.replace('+', ' '), acc * 100))
        
        results[name] = acc

    print('summary: {:.2f}%'.format(ave_acc / ave_acc_n * 100))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
