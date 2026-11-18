"""Convert locally obtained benchmark data to the formats used by MoEE."""

import argparse
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.data.utils import save_graphs


DATASETS = (
    "romanempire", "amazonratings", "blogcatalog", "flickr", "photo",
    "wikics", "penn94", "genius", "snappatents", "ogbnarxiv",
)
PARTS = ("train", "val", "test")
N_RUNS = 10


def mask_from_indices(indices, num_nodes):
    mask = np.zeros(num_nodes, dtype=np.bool_)
    mask[indices] = True
    return mask


def repeat_masks(indices, num_nodes):
    return {
        name: np.repeat(mask_from_indices(idx, num_nodes)[None, :], N_RUNS, axis=0)
        for name, idx in zip(PARTS, indices)
    }


def random_masks(labels, seed):
    rng = np.random.RandomState(seed)
    eligible = np.flatnonzero(labels >= 0)
    first = len(eligible) * 48 // 100
    second = len(eligible) * 80 // 100
    splits = {name: [] for name in PARTS}
    for _ in range(N_RUNS):
        for name, idx in zip(PARTS, np.split(rng.permutation(eligible), [first, second])):
            splits[name].append(mask_from_indices(idx, len(labels)))
    return {name: np.stack(rows) for name, rows in splits.items()}


def load_arrays(path, split_seed):
    with np.load(str(path), allow_pickle=False) as data:
        edges = data["edges"]
        features = data["node_features"]
        labels = data["node_labels"].reshape(-1)
        keys = [name + "_masks" for name in PARTS]
        present = [key in data for key in keys]
        if all(present) and split_seed is None:
            masks = {name: data[key] for name, key in zip(PARTS, keys)}
        elif not any(present) and split_seed is not None:
            masks = random_masks(labels, split_seed)
        else:
            raise ValueError("Supply all three mask arrays or omit all three and set --split-seed.")
    return edges, features, labels, masks


def load_ogb(root):
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(root))
    graph, labels = dataset[0]
    split = dataset.get_idx_split()
    labels = labels.reshape(-1)
    masks = repeat_masks([split[key] for key in ("train", "valid", "test")], len(labels))
    return graph["edge_index"].T, graph["node_feat"], labels, masks


def validate(edges, features, labels, masks):
    n = len(labels)
    if features.ndim != 2 or features.shape[0] != n:
        raise ValueError("node_features must have shape [number of nodes, number of features].")
    if labels.dtype.kind not in "iu" or (labels < -1).any():
        raise ValueError("node_labels must be integer class IDs, or -1 for unlabelled nodes.")
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("edges must be an integer array of shape [number of edges, 2].")
    if (edges < 0).any() or (edges >= n).any():
        raise ValueError("Edge indices must be zero-based node IDs.")
    for name in PARTS:
        mask = masks[name]
        if mask.shape != (N_RUNS, n) or mask.dtype.kind != "b" or not mask.any(1).all():
            raise ValueError("{}_masks must contain ten nonempty Boolean masks.".format(name))
        if mask[:, labels < 0].any():
            raise ValueError("The split includes unlabelled nodes.")
    a, b, c = [masks[name] for name in PARTS]
    if ((a & b) | (a & c) | (b & c)).any():
        raise ValueError("Train, validation and test masks overlap within at least one run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input", type=Path, help="Local NPZ, not used for ogbnarxiv.")
    parser.add_argument("--split-seed", type=int, help="Generate substitute splits with this seed.")
    parser.add_argument("--ogb-root", type=Path, default=Path("raw/ogb"))
    args = parser.parse_args()
    if args.dataset == "ogbnarxiv":
        if args.input is not None or args.split_seed is not None:
            parser.error("ogbnarxiv uses the official OGB download and split.")
        edges, features, labels, masks = load_ogb(args.ogb_root)
    else:
        if args.input is None:
            parser.error("--input is required for non-OGB datasets.")
        edges, features, labels, masks = load_arrays(args.input, args.split_seed)
    validate(edges, features, labels, masks)

    graph_path = Path("data/graphs") / (args.dataset + ".pt")
    split_path = Path("data/splits") / (args.dataset + "_splits.pt")
    if graph_path.exists() or split_path.exists():
        parser.error("Output already exists. Move the old files before converting again.")
    for output in (graph_path, split_path):
        output.parent.mkdir(parents=True, exist_ok=True)

    graph = dgl.graph(
        (torch.as_tensor(edges[:, 0], dtype=torch.long), torch.as_tensor(edges[:, 1], dtype=torch.long)),
        num_nodes=len(labels),
    )
    if args.dataset in ("romanempire", "amazonratings"):
        graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = torch.as_tensor(features, dtype=torch.float32)
    graph.ndata["label"] = torch.as_tensor(labels, dtype=torch.long)
    save_graphs(str(graph_path), [graph])
    torch.save({name: torch.from_numpy(mask) for name, mask in masks.items()}, str(split_path))
    print("Wrote {} and {}".format(graph_path, split_path))


if __name__ == "__main__":
    main()
