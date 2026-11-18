import argparse
import gc
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from module.moe import MoEE
from utils import (
    compute_edge_ppr_gpu,
    load_data,
    load_fixed_data_split,
    seed_everything,
    set_device,
    setup_cfg,
    str2bool,
    tab_printer,
)


def summarize(values):
    mean = float(np.mean(values)) * 100.0
    std = float(np.std(values)) * 100.0
    print("ACC mean +/- std: {:.2f} +/- {:.2f}".format(mean, std))
    return mean, std


def run(args):
    device = set_device(args.device)
    graph, labels, num_classes = load_data(
        dataset_name=args.dataset,
        normalize=args.normalize,
        undirected=args.undirected,
        self_loop=args.self_loop,
    )

    args.ppr_path = compute_edge_ppr_gpu(
        adj_sp=graph.adj(scipy_fmt="csr"),
        dataset=args.dataset,
        cache_root=args.ppr_cache,
        alpha=args.ppr_alpha,
        max_hops=args.ppr_hops,
        walks_per_node=args.ppr_walks,
        source_batch_size=args.ppr_batch_size,
        seed=args.ppr_seed,
        device=device,
    )

    seed_everything(args.seed)
    run_seeds = [random.randint(0, 99999) for _ in range(args.runs)]
    scores = []
    start = time.time()

    for run_id, run_seed in enumerate(run_seeds):
        print("\nRun: {} (seed={})\n".format(run_id, run_seed))
        seed_everything(run_seed)
        train_mask, val_mask, test_mask = load_fixed_data_split(
            args.dataset, run_id
        )

        model = MoEE(
            graph.ndata["feat"].shape[1],
            num_classes,
            device,
            args,
        )
        model.fit(graph, labels, train_mask, val_mask, test_mask)
        prediction, _, _, _ = model.predict(graph)
        score = accuracy_score(labels[test_mask], prediction[test_mask])
        scores.append(score)
        print("Test ACC: {:.6f}".format(score))

        del model, prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTotal cost: {:.4f}s".format(time.time() - start))
    return summarize(scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the main MoEE node-classification experiment."
    )
    parser.add_argument("--config", default="./config/MoEE.yaml")
    parser.add_argument("--dataset", default="romanempire")
    parser.add_argument("--device", default="0", help="GPU id or 'none'")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2_coef", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--undirected", type=str2bool, default=False)
    parser.add_argument("--self_loop", type=str2bool, default=False)
    parser.add_argument("--normalize", type=int, default=-1)
    parser.add_argument("--nhidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--r", type=float, default=None,
        help="Override the Prune budget in the dataset configuration.",
    )
    parser.add_argument("--router_tau", type=float, default=1.0)
    parser.add_argument("--ppr_cache", default="./save/pprs")
    parser.add_argument("--ppr_alpha", type=float, default=0.15)
    parser.add_argument("--ppr_hops", type=int, default=32)
    parser.add_argument("--ppr_walks", type=int, default=512)
    parser.add_argument("--ppr_batch_size", type=int, default=4096)
    parser.add_argument("--ppr_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    r_override = arguments.r
    setup_cfg(arguments, vars(arguments))
    if r_override is not None:
        arguments.r = r_override
    if arguments.runs < 1 or arguments.runs > 10:
        raise ValueError("--runs must be between 1 and 10 (ten splits are provided)")
    tab_printer(vars(arguments))
    run(arguments)
