# ARGCN-DKG

## Requirements

* python 3
* pytorch 0.4.0
* nltk

## Instructions

### Materials Preparation

There is a folder `materials/`, which contains some meta data and programs already.

#### Feature and Word Embedding
1. Download: http://datasets.d2.mpi-inf.mpg.de/xian/ILSVRC2012_res101_feature.mat
2. Download: http://datasets.d2.mpi-inf.mpg.de/xian/ImageNet2011_res101_feature.zip
3. Download: https://github.com/pujols/zero-shot-learning/blob/master/ImageNet_w2v.zip

#### Graphs
1. `cd materials/`
2. Run `python make_induced_graph.py`, get `imagenet-induced-graph.json`, for WordNet
3. Run `python make_relation_graph.py`, get `imagenet-relation-graph.json`, for Wup

#### Image Feature Extraction
1. `cd materials/`
2. Run `python process_resnet101.py`, get `fc-weights.json`

### Training

Make a directory `save/` for saving models.

In most programs, use `--gpu` to specify the devices to run the code (default: use gpu 0).

#### Train GCN
* GCN(for WordNet Graph): Run `python train_gcn_basic.py --no-residual`, get results in `save/gcn-basic`
* GCN(for Wup Graph): Run `python train_gcn_relation.py --no-residual`, get results in `save/gcn_relation`
* RGCN(for WordNet Graph): Run `python train_gcn_basic.py`, get results in `save/gcn-basic`
* RGCN(for Wup Graph): Run `python train_gcn_relation.py`, get results in `save/gcn-relation`
* ARGCN(for WordNet Graph): Run `python train_gat_basic.py`, get results in `save/gat-basic`
* ARGCN(for Wup Graph): Run `python train_gat_relation.py`, get results in `save/gat-relation`

In the results folder:
* `*.pth` is the state dict of GCN model
* `*.pred` is the prediction file, which can be loaded by `torch.load()`. It is a python dict, having two keys: `wnids` - the wordnet ids of the predicted classes, `pred` - the predicted fc weights

### Testing

#### ImageNet
Run `python evaluate_imagenet.py` with the args:
* `--alpha`: the weight for WordNet and Wup ensemble, alpha * WordNet + (1 - alpha) * Wup
* `--pred`: the `.pred` file for testing, can use multi files for ensemble, separated by space
* `--test-set`: load test set in `materials/imagenet-testsets.json`, choices: `[2-hops, 3-hops, all]`
* (optional) `--keep-ratio` for the ratio of testing data, `--consider-trains` to include training classes' classifiers, `--test-train` for testing with train classes images only.
