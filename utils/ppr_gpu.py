import hashlib
import math
import os
from pathlib import Path
import time


import numpy as np
import scipy.sparse as sp
import torch


PPR_CACHE_VERSION = 2


def _prepare_binary_csr(adj_sp):
    """Return a canonical, sorted CSR adjacency and its row transition probs."""
    adj = sp.csr_matrix(adj_sp, dtype=np.float32)
    adj.sum_duplicates()
    adj.eliminate_zeros()
    adj.sort_indices()

    if adj.nnz == 0:
        raise ValueError("The graph has no edges.")
    if np.any(adj.data <= 0):
        raise ValueError("PPR sampling requires positive adjacency weights.")

    # Current benchmarks are expected to be binary/unweighted graphs.
    first = float(adj.data[0])
    if not np.allclose(adj.data, first, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "Non-uniform edge weights detected. Implement weighted CSR sampling "
            "instead of uniform-neighbor sampling."
        )

    indptr = adj.indptr.astype(np.int64, copy=False)
    indices = adj.indices.astype(np.int64, copy=False)
    degree = np.diff(indptr)

    transition_prob = np.zeros(adj.nnz, dtype=np.float32)
    nonzero_rows = degree > 0
    inv_degree = np.zeros(adj.shape[0], dtype=np.float32)
    inv_degree[nonzero_rows] = 1.0 / degree[nonzero_rows]
    src = np.repeat(np.arange(adj.shape[0], dtype=np.int64), degree)
    transition_prob[:] = inv_degree[src]
    return adj, indptr, indices, src, transition_prob


def _csr_fingerprint(adj):
    """Fingerprint graph structure and values to prevent stale-cache reuse."""
    h = hashlib.sha256()
    h.update(np.asarray(adj.shape, dtype=np.int64).tobytes())
    h.update(adj.indptr.astype(np.int64, copy=False).tobytes())
    h.update(adj.indices.astype(np.int64, copy=False).tobytes())
    h.update(adj.data.astype(np.float32, copy=False).tobytes())
    return h.hexdigest()


@torch.inference_mode()
def _sample_uniform_neighbor(nodes, indptr, indices, generator):
    """Sample one outgoing neighbor for every node in `nodes` on GPU."""
    # -1 is a cemetery state. This matches the current zero-row treatment of
    # D^{-1}A: once a walk reaches a zero-outdegree node, its remaining mass
    # does not propagate through A. Do not silently turn dangling nodes into
    # absorbing self-loops unless the paper explicitly adopts that convention.
    out = torch.full_like(nodes, -1)
    live = nodes >= 0
    starts = torch.zeros_like(nodes)
    degrees = torch.zeros_like(nodes)
    starts[live] = indptr[nodes[live]]
    degrees[live] = indptr[nodes[live] + 1] - starts[live]

    valid = live & (degrees > 0)
    if valid.any():
        valid_starts = starts[valid]
        valid_degrees = degrees[valid]
        random_offset = torch.floor(
            torch.rand(
                valid_degrees.numel(),
                device=nodes.device,
                generator=generator,
            ) * valid_degrees.to(torch.float32)
        ).to(torch.long)
        out[valid] = indices[valid_starts + random_offset]
    return out


@torch.inference_mode()
def _residual_edge_hits(
    source_begin,
    source_end,
    walks_per_node,
    max_hops,
    alpha,
    num_nodes,
    indptr,
    indices,
    edge_hash,
    generator,
):
    """Return edge positions hit by the >=2-hop residual random walks."""
    sources = torch.arange(
        source_begin, source_end, device=indptr.device, dtype=torch.long
    )
    origins = torch.repeat_interleave(sources, walks_per_node)

    # Conditional on path length >= 2, the first two transitions are certain.
    current = _sample_uniform_neighbor(origins, indptr, indices, generator)
    current = _sample_uniform_neighbor(current, indptr, indices, generator)
    alive = current >= 0

    # At lengths 2,...,K-1, stop with alpha or continue with 1-alpha.
    for _length in range(2, max_hops):
        continue_now = alive & (
            torch.rand(
                alive.numel(), device=alive.device, generator=generator
            ) >= alpha
        )
        if not continue_now.any():
            break
        current[continue_now] = _sample_uniform_neighbor(
            current[continue_now], indptr, indices, generator
        )
        alive = continue_now

    valid_endpoint = current >= 0
    origins = origins[valid_endpoint]
    current = current[valid_endpoint]
    query_hash = origins * num_nodes + current
    positions = torch.searchsorted(edge_hash, query_hash)
    positions_safe = positions.clamp(max=edge_hash.numel() - 1)
    valid = (positions < edge_hash.numel()) & (
        edge_hash[positions_safe] == query_hash
    )
    return positions_safe[valid]


@torch.no_grad()
def _descending_csr_ranks(src, score):
    """
    Rank edges within each source row.
    Rank 1 means the largest PPR.

    Compatible with old PyTorch. Equal PPR values share the same
    competition rank, so stable sorting is unnecessary.
    """
    # Different source rows occupy disjoint key ranges.
    # Within one row, larger PPR produces a smaller key.
    key = src.to(torch.float64) * 2.0 - score.to(torch.float64)

    order = torch.argsort(key, dim=0, descending=False)
    sorted_src = src[order]
    sorted_score = score[order]

    position = torch.arange(
        src.numel(), device=src.device, dtype=torch.long
    )

    # Starting position of each source row.
    is_new_row = torch.ones_like(sorted_src, dtype=torch.bool)
    is_new_row[1:] = sorted_src[1:] != sorted_src[:-1]

    row_start = torch.where(
        is_new_row, position, torch.zeros_like(position)
    )
    row_start = torch.cummax(row_start, dim=0)[0]

    # Equal scores in the same row receive the same rank.
    is_new_value = is_new_row.clone()
    is_new_value[1:] = (
        is_new_value[1:]
        | (sorted_score[1:] != sorted_score[:-1])
    )

    value_start = torch.where(
        is_new_value, position, torch.zeros_like(position)
    )
    value_start = torch.cummax(value_start, dim=0)[0]

    # Competition ranking: e.g. 1, 1, 3.
    sorted_rank = value_start - row_start + 1

    rank = torch.empty(
        src.numel(), device=src.device, dtype=torch.int32
    )
    rank[order] = sorted_rank.to(torch.int32)
    return rank


@torch.inference_mode()
def compute_edge_ppr_gpu(
    adj_sp,
    dataset,
    cache_root="./save/pprs",
    alpha=0.15,
    max_hops=8,
    walks_per_node=64,
    source_batch_size=32768,
    seed=42,
    device="cuda",
):
    """
    Approximate K-step truncated PPR only on original edges.

    max_hops must be in [1, 32]. Peak temporary memory is controlled by
    source_batch_size; on CUDA OOM the batch size is halved and retried.
    """
    time_0 = time.time()
    if not 1 <= int(max_hops) <= 32:
        raise ValueError(f"max_hops must be in [1, 32], got {max_hops}")
    if walks_per_node <= 0:
        raise ValueError("walks_per_node must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device)

    adj, indptr_np, indices_np, src_np, transition_np = _prepare_binary_csr(
        adj_sp
    )
    fingerprint = _csr_fingerprint(adj)
    n = adj.shape[0]
    e = adj.nnz

    cache_dir = Path(cache_root) / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    alpha_name = str(alpha).replace(".", "p")
    cache_path = cache_dir / (
        f"ppr_mc_v{PPR_CACHE_VERSION}_a{alpha_name}_k{max_hops}_"
        f"w{walks_per_node}_s{seed}_{fingerprint[:12]}.pt"
    )
    if cache_path.exists():
        print(f"Reuse PPR cache: {cache_path}")
        return str(cache_path)

    print(
        f"Compute edge PPR: N={n:,}, E={e:,}, alpha={alpha}, "
        f"max_hops={max_hops}, walks={walks_per_node}, device={device}"
    )

    indptr = torch.from_numpy(indptr_np).to(device=device, dtype=torch.long)
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    src = torch.from_numpy(src_np).to(device=device, dtype=torch.long)
    transition = torch.from_numpy(transition_np).to(
        device=device, dtype=torch.float32
    )
    edge_hash = src * n + indices

    # Exact 0-hop and 1-hop terms.
    ppr = torch.zeros(e, device=device, dtype=torch.float32)
    self_loop = src == indices
    ppr[self_loop] += alpha

    if max_hops == 1:
        ppr += (1.0 - alpha) * transition
    else:
        ppr += alpha * (1.0 - alpha) * transition

        # The total mass for paths of length >=2 is (1-alpha)^2.
        residual_mass = (1.0 - alpha) ** 2
        hit_count = torch.zeros(e, device=device, dtype=torch.int32)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        source_begin = 0
        current_batch_size = min(int(source_batch_size), n)
        while source_begin < n:
            source_end = min(source_begin + current_batch_size, n)
            rng_state = generator.get_state()
            try:
                hit_positions = _residual_edge_hits(
                    source_begin=source_begin,
                    source_end=source_end,
                    walks_per_node=walks_per_node,
                    max_hops=max_hops,
                    alpha=alpha,
                    num_nodes=n,
                    indptr=indptr,
                    indices=indices,
                    edge_hash=edge_hash,
                    generator=generator,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if current_batch_size <= 256:
                    raise
                # Retry the same source range without consuming a different
                # random-number subsequence.
                generator.set_state(rng_state)
                current_batch_size //= 2
                torch.cuda.empty_cache()
                print(f"CUDA OOM: retry with source_batch_size={current_batch_size}")
                continue

            if hit_positions.numel() > 0:
                hit_count.scatter_add_(
                    0,
                    hit_positions,
                    torch.ones_like(hit_positions, dtype=torch.int32),
                )
            source_begin = source_end
            if source_begin % max(current_batch_size * 10, 1) == 0:
                print(f"PPR sources: {source_begin:,}/{n:,}")

        ppr += residual_mass * hit_count.to(torch.float32) / walks_per_node
        del hit_count

    rank = _descending_csr_ranks(src, ppr)

    result = {
        "ppr_value": ppr.cpu(),
        "ppr_rank": rank.cpu(),
        "meta": {
            "version": PPR_CACHE_VERSION,
            "solver": "gpu_batched_monte_carlo_truncated_ppr",
            "alpha": float(alpha),
            "max_hops": int(max_hops),
            "walks_per_node": int(walks_per_node),
            "seed": int(seed),
            "num_nodes": int(n),
            "num_edges": int(e),
            "graph_fingerprint": fingerprint,
            "rank_order": "descending_ppr_then_destination_id",
            "edge_order": "canonical_csr_row_then_column",
        },
    }
    torch.save(result, cache_path)
    print(f"Saved PPR cache: {cache_path}")
    print("PPR cost: {:.4f}s".format(time.time() - time_0))
    return str(cache_path)
