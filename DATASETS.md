# Data preparation

Obtain the data from the original providers below. The repository does not
redistribute any graph, splits, or cache files. Use the files under the
provider's terms.

| CLI name | Original source | Graph inputs |
|---|---|---|
| `romanempire` | [Yandex Research](https://github.com/yandex-research/heterophilous-graphs/tree/main/data) | `roman_empire.npz` |
| `amazonratings` | [Yandex Research](https://github.com/yandex-research/heterophilous-graphs/tree/main/data) | `amazon_ratings.npz` |
| `blogcatalog`, `flickr` | [CoLA](https://github.com/TrustAGI-Lab/CoLA/tree/main/raw_dataset) | Adjacency, attributes, and the multi-class group labels, not anomaly flags |
| `photo` | [gnn-benchmark](https://github.com/shchur/gnn-benchmark/tree/master/data/npz) | Amazon Photo adjacency, features, and labels |
| `wikics` | [Wiki-CS](https://github.com/pmernyei/wiki-cs-dataset/tree/master/dataset) | `links`, `features`, and `labels` from the data JSON |
| `penn94`, `genius`, `snappatents` | [LINKX data loader](https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/dataset.py) and [dataset files](https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data) | Use each upstream graph and its node features and labels. For Penn94, the upstream `fb100` loader with sub-dataset `Penn94` provides the one-hot node features and the binary label. |
| `ogbnarxiv` | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | OGB graph and official time split |

## Convert locally

Run commands from the repository root. For Yandex, pass the downloaded
NPZ directly to the converter. It uses the published masks in that file
and builds the bidirectional graph following the upstream loader.
For Roman-Empire, the upstream masks differ from the fixed masks used
for the paper, so this example follows the upstream splits.

```sh
python prepare_data.py --dataset amazonratings --input raw/amazon_ratings.npz
```

For ogbn-arxiv, install the official `ogb` package compatible with your Python
environment and run:

```sh
python prepare_data.py --dataset ogbnarxiv
```

This downloads the graph through OGB, preserves the node order and the
official train/validation/test indices, and repeats the masks for ten runs.
The training configuration adds reverse edges and merges duplicates.

For other datasets, use the original loader to obtain the node arrays,
then export a local NumPy NPZ with these numeric fields (no objects):

- `edges`: integer array of shape `[E, 2]`. Each row is a zero-based
  source/destination pair. For adjacency matrices use their nonzero row
  and column indices. Preserve the source graph's edge direction.
- `node_features`: dense numeric matrix of shape `[N, F]` (convert sparse
  attributes to a dense array if needed).
- `node_labels`: one integer class ID per node, with classes starting at
  zero and `-1` for nodes without a label. Convert a one-hot group array
  to class IDs before saving.
- `train_masks`, `val_masks`, `test_masks`: Boolean arrays of shape
  `[10, N]` for fixed splits, with one row per run. Keep the node ordering
  identical across edges, features, labels, and masks.

```sh
python prepare_data.py --dataset photo --input raw/photo_arrays.npz
```

If you do not have the experiment masks, leave all three mask fields out
and pass `--split-seed 42` to generate ten random 48%/32%/20% splits over
the labeled nodes. These are a reproducible substitute, not a claim to
recover the exact masks or scores reported in the paper. Likewise, do not
assume a provider's alternative split protocol matches the paper.

The converter writes `data/graphs/{dataset}.pt` (a DGL graph with node
features `feat` and labels `label`) and
`data/splits/{dataset}_splits.pt` (a Torch dictionary with `train`, `val`,
and `test` masks). If you already generated these files, you can place them
there instead. The training loader applies the graph options in the
experiment configuration and merges duplicate edges into unweighted pairs.
