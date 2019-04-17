import json
import h5py
import numpy as np
from tqdm import trange

def get_feature_avg(ilsvrc_path, wnid_path):
    wnid_feat_dict = {}
    wnid_cnt_dict = {}
    data = h5py.File(ilsvrc_path)
    features = np.array(data['features']) # 2048 * 1281167
    print('features shape:', features.shape)
    features_val = np.array(data['features_val']).T # 2048 * 50000
    print('features_val shape:', features_val.shape)
    labels = np.array(data['labels'][0]) # 1281167
    print('labels shape:', labels.shape)
    labels_val = np.array(data['labels_val'][0]) # 50000
    print('labels_val shape:', labels_val.shape)
    suppl = h5py.File(wnid_path)
    wnids = [''.join(chr(i) for i in suppl[wnid[0]]) for wnid in suppl['wnids'][:1000]]
    for ii in trange(len(labels)):
        wnid = wnids[int(labels[ii])-1]
        feature = features[:, ii]
        if wnid not in wnid_feat_dict:
            wnid_feat_dict[wnid] = np.zeros((feature.shape[0]))
            wnid_cnt_dict[wnid] = 0
        wnid_feat_dict[wnid] += feature
        wnid_cnt_dict[wnid] += 1
    for ii in trange(len(labels_val)):
        wnid = wnids[int(labels_val[ii])-1]
        feature = features_val[:, ii]
        if wnid not in wnid_feat_dict:
            wnid_feat_dict[wnid] = np.zeros((feature.shape[0]))
            wnid_cnt_dict[wnid] = 0
        wnid_feat_dict[wnid] += feature
        wnid_cnt_dict[wnid] += 1
    for wnid in wnid_feat_dict:
        wnid_feat_dict[wnid] /= wnid_cnt_dict[wnid]
    print('wnid_feature_dict length:', len(wnid_feat_dict))
    return wnids, wnid_feat_dict

wnids, wnid_feat_dict = get_feature_avg('/home/t-yud/ZSL/data/ImageNet/ILSVRC2012_res101_feature.mat', '/home/t-yud/ZSL/data/ImageNet/ImageNet_w2v/ImageNet_w2v.mat')

# wnids = json.load(open('imagenet-split.json', 'r'))['train']
# wnids = sorted(wnids)
obj = []
for wnid in wnids:
    obj.append((wnid, wnid_feat_dict[wnid].tolist()))
json.dump(obj, open('fc-weights.json', 'w'))

