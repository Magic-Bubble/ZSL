import argparse
import json

from nltk.corpus import wordnet as wn
import torch
from torch.nn import functional as F

import h5py
import numpy as np
from tqdm import tqdm

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def getedges(s, topk):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in tqdm(enumerate(s)):
        queue = [u]
        seen = set([u])
        k = 0
        while k < len(queue) and len(queue) < 20:
            x = queue[k]
            for v in x.hypernyms() + x.hyponyms():
                if v not in seen:
                    j = dic.get(v)
                    if j is not None:
                        seen.add(v)
                        queue.append(v)
            k += 1
        sim = [u.wup_similarity(q) for q in queue[:20]]
        topsim = np.argsort(-np.array(sim))[:topk]
        edges.extend([(i, dic[queue[q]]) for q in topsim])
    edges = list(set(edges))
    return edges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='imagenet-split.json')
    parser.add_argument('--output', default='imagenet-relation-graph.json')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    print('using pretrained w2v of xian ...')

    suppl = h5py.File('/home/t-yud/ZSL/data/ImageNet/ImageNet_w2v/ImageNet_w2v.mat')
    wnids = [''.join(chr(i) for i in suppl[wnid[0]]) for wnid in suppl['wnids']]
    nodes = list(map(getnode, wnids))
    vectors = np.array(suppl['w2v']).T
    print(len(wnids), vectors.shape)

    print('making graph ...')

    edges = getedges(nodes, topk=args.topk)
    print(len(edges), edges[0])

    print('dumping ...')

    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()
    obj['edges'] = edges
    json.dump(obj, open(args.output, 'w'))

