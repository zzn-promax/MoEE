# MoEE

Implementation and experiment settings for **From Nodes to Edges: A
Mixture-of-Edge-Experts Framework for Graphs** (ACM TKDD).

This repository contains the MoEE model, edge router, PPR preprocessing,
main node-classification runner, and selected configurations for all ten
datasets. It does not ship dataset files, split files, PPR caches, or
third-party baseline code. See [data preparation](DATASETS.md) and
[baseline papers and official code](BASELINES.md).

## Environment

The main experiments used Python 3.6.13, PyTorch 1.10.2 with CUDA 11.3,
DGL 0.8.1, and torch-scatter 2.0.9. The dependency versions for this
runner are listed in [requirements-tested.txt](requirements-tested.txt).
Install PyTorch, DGL, and torch-scatter builds compatible with the
local Python and CUDA versions before installing the other packages.

## Run

From the repository root, first prepare the graph and splits as shown in
[DATASETS.md](DATASETS.md). For example, after saving the original Yandex
Roman-Empire archive as `raw/roman_empire.npz`:

```sh
python prepare_data.py --dataset romanempire --input raw/roman_empire.npz
python main_MoEE.py --dataset romanempire --device 0 --runs 1 --epochs 2
python main_MoEE.py --dataset romanempire --device 0
```

The final command runs ten seeds with the dataset-specific settings in
[`config/MoEE.yaml`](config/MoEE.yaml). The first run computes its own
edge-level PPR cache and stores it locally. The default PPR settings are
restart probability 0.15, 512 walks per source, at most 32 steps per walk,
an initial batch of 4,096 sources, and seed 42. Use `--r` to override the
dataset's configured Prune budget.

The official ogbn-arxiv time split is used for every run on that dataset.
For other datasets, use the exact experiment masks if available. The
preparation guide distinguishes published masks and newly generated splits;
results with different splits need not match the paper's table.
