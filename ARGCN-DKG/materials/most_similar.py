
import numpy as np
import json
from nltk.corpus import wordnet as wn

def get_data(json_file):
    data = json.load(open(json_file, 'r'))
    vec = np.array(data['vectors'])
    wnid = data['wnids']
    return vec, wnid
embed, wnids = get_data('./imagenet-induced-graph.json')

def getnode(wnid):
    node = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
    return node

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

def cosine_sim(vec1, vec2):
    num = (vec1 * vec2).sum()
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
    cos = num / denom
    return cos

def most_similar(wnid, topk=10):
    print('search most similar word vector for', getnode(wnid))
    sim_dict = {}
    idx = wnids.index(wnid)
    print(idx)
    for ii, emb in enumerate(embed):
        sim_dict[getnode(wnids[ii])] = cosine_sim(embed[idx], emb)
    sim = list(sorted(sim_dict.items(), key=lambda x: x[1], reverse=True))
    print(sim[:topk])
    
if __name__ == '__main__':
    most_similar('n03664943', topk=20)
