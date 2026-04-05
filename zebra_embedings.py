import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

import torch
import torch.nn as nn

from numba import njit

# =====================================================
# PATHS
# =====================================================

GRAPH_PATH = os.path.expanduser("~/.cache/RP/hacker_subgraph_k1_T.parquet")
SORTED_GRAPH_PATH = GRAPH_PATH.replace(".parquet", "_sorted.parquet")
OUT_EMB_PATH = os.path.expanduser("~/.cache/RP/zebra_node_embeddings.parquet")

SRC_COL = "addr_id1"
DST_COL = "addr_id2"
TIME_COL = "height"
AMOUNT_COL = "amount"

# =====================================================
# SETTINGS
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

TOP_K = 20
ENSEMBLE = [
    (0.3, 0.5),
    (0.3, 0.95),
]

NODE_DIM = 32
TIME_DIM = 8
AMOUNT_DIM = 8
EMB_DIM = 32

CHUNK_SIZE = 1_000_000
MAX_TOTAL_EDGES = 10_000_000

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =====================================================
# SORT / STREAM
# =====================================================

def check_sorted(path):
    pf = pq.ParquetFile(path)
    last = None
    for batch in pf.iter_batches(batch_size=1_000_000, columns=[TIME_COL]):
        ts = np.array(batch.to_pydict()[TIME_COL], dtype=np.int64)
        if len(ts) == 0:
            continue
        if last is not None and ts[0] < last:
            return False
        last = ts[-1]
    return True


def sort_graph_if_needed():
    global GRAPH_PATH

    if os.path.exists(SORTED_GRAPH_PATH) and check_sorted(SORTED_GRAPH_PATH):
        GRAPH_PATH = SORTED_GRAPH_PATH
        print("Using sorted file:", GRAPH_PATH)
        return

    if check_sorted(GRAPH_PATH):
        print("Current file already sorted")
        return

    print("Sorting graph...")
    df = pd.read_parquet(GRAPH_PATH)
    df = df.sort_values([TIME_COL], kind="mergesort")
    df.to_parquet(SORTED_GRAPH_PATH)
    GRAPH_PATH = SORTED_GRAPH_PATH
    print("Saved sorted graph:", GRAPH_PATH)


def iter_edges(path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=[SRC_COL, DST_COL, TIME_COL, AMOUNT_COL],
    ):
        d = batch.to_pydict()
        yield (
            np.array(d[SRC_COL], dtype=np.int64),
            np.array(d[DST_COL], dtype=np.int64),
            np.array(d[TIME_COL], dtype=np.int64),
            np.log1p(np.array(d[AMOUNT_COL], dtype=np.float32)),
        )


def iter_edges_grouped_by_time(path, max_total_edges=None):
    seen = 0
    current_t = None
    group = []

    for src, dst, ts, amt in iter_edges(path):
        for u, v, t, a in zip(src, dst, ts, amt):
            if max_total_edges is not None and seen >= max_total_edges:
                if group:
                    yield group
                return

            u = int(u)
            v = int(v)
            t = int(t)
            a = float(a)

            if current_t is None:
                current_t = t

            if t != current_t:
                yield group
                group = []
                current_t = t

            group.append((u, v, t, a))
            seen += 1

    if group:
        yield group


# =====================================================
# SANTA
# =====================================================

@njit
def _insert_desc_temporal(node_ids, times, amts, vals, size, k,
                          new_node, new_time, new_amt, new_val):
    for i in range(size):
        if node_ids[i] == new_node and times[i] == new_time:
            vals[i] += new_val

            j = i
            while j > 0 and vals[j] > vals[j - 1]:
                node_ids[j], node_ids[j - 1] = node_ids[j - 1], node_ids[j]
                times[j], times[j - 1] = times[j - 1], times[j]
                amts[j], amts[j - 1] = amts[j - 1], amts[j]
                vals[j], vals[j - 1] = vals[j - 1], vals[j]
                j -= 1
            return size

    if size < k:
        node_ids[size] = new_node
        times[size] = new_time
        amts[size] = new_amt
        vals[size] = new_val
        size += 1

        j = size - 1
        while j > 0 and vals[j] > vals[j - 1]:
            node_ids[j], node_ids[j - 1] = node_ids[j - 1], node_ids[j]
            times[j], times[j - 1] = times[j - 1], times[j]
            amts[j], amts[j - 1] = amts[j - 1], amts[j]
            vals[j], vals[j - 1] = vals[j - 1], vals[j]
            j -= 1
        return size

    if new_val <= vals[k - 1]:
        return size

    node_ids[k - 1] = new_node
    times[k - 1] = new_time
    amts[k - 1] = new_amt
    vals[k - 1] = new_val

    j = k - 1
    while j > 0 and vals[j] > vals[j - 1]:
        node_ids[j], node_ids[j - 1] = node_ids[j - 1], node_ids[j]
        times[j], times[j - 1] = times[j - 1], times[j]
        amts[j], amts[j - 1] = amts[j - 1], amts[j]
        vals[j], vals[j - 1] = vals[j - 1], vals[j]
        j -= 1

    return size


@njit
def update_reservoir_numba(
    nodes_u, times_u, amts_u, vals_u, len_u,
    nodes_v, times_v, amts_v, vals_v, len_v,
    m_u, alpha, beta, k,
    new_node, new_time, new_amt
):
    denom = m_u * beta + beta
    c1 = (m_u * beta) / denom
    c2 = (beta * (1.0 - alpha)) / denom

    out_nodes = np.full(k, -1, dtype=np.int64)
    out_times = np.full(k, -1, dtype=np.int64)
    out_amts = np.zeros(k, dtype=np.float32)
    out_vals = np.zeros(k, dtype=np.float32)
    out_len = 0

    for i in range(len_u):
        if nodes_u[i] != -1 and vals_u[i] > 0:
            out_len = _insert_desc_temporal(
                out_nodes, out_times, out_amts, out_vals, out_len, k,
                nodes_u[i], times_u[i], amts_u[i], c1 * vals_u[i]
            )

    for i in range(len_v):
        if nodes_v[i] != -1 and vals_v[i] > 0:
            out_len = _insert_desc_temporal(
                out_nodes, out_times, out_amts, out_vals, out_len, k,
                nodes_v[i], times_v[i], amts_v[i], c2 * vals_v[i]
            )

    out_len = _insert_desc_temporal(
        out_nodes, out_times, out_amts, out_vals, out_len, k,
        new_node, new_time, new_amt, c2 * alpha
    )

    s = 0.0
    for i in range(out_len):
        s += out_vals[i]

    if s > 0:
        inv = 1.0 / s
        for i in range(out_len):
            out_vals[i] *= inv

    return out_nodes, out_times, out_amts, out_vals, out_len


@njit
def update_m_numba(m_val, beta):
    return m_val * beta + beta


# =====================================================
# ZEBRA ENCODER
# =====================================================

class ZebraEncoder(nn.Module):
    def __init__(self, max_node_id):
        super().__init__()

        self.node_emb = nn.Embedding(max_node_id + 1, NODE_DIM)
        self.time_enc = nn.Linear(1, TIME_DIM)
        self.amount_enc = nn.Linear(1, AMOUNT_DIM)

        self.transform = nn.Sequential(
            nn.Linear(NODE_DIM + TIME_DIM + AMOUNT_DIM, EMB_DIM),
            nn.ReLU(),
            nn.Linear(EMB_DIM, EMB_DIM),
        )

    def zebra_embed_single(self, reservoir_tuple, current_time):
        if reservoir_tuple is None:
            return torch.zeros(EMB_DIM, device=DEVICE)

        node_ids, times, amts, vals, size = reservoir_tuple

        if size == 0:
            return torch.zeros(EMB_DIM, device=DEVICE)

        neighs = torch.tensor(node_ids[:size], dtype=torch.long, device=DEVICE)
        taus = times[:size]
        edge_amts = amts[:size]

        weights = torch.tensor(vals[:size], dtype=torch.float32, device=DEVICE)
        weights = weights / (weights.sum() + 1e-8)

        node_feat = self.node_emb(neighs)

        dt_vals = [[max(int(current_time) - int(tau), 0)] for tau in taus]
        dt = torch.tensor(dt_vals, dtype=torch.float32, device=DEVICE)
        dt = torch.log1p(dt)

        amt_vals = [[float(a)] for a in edge_amts]
        amt = torch.tensor(amt_vals, dtype=torch.float32, device=DEVICE)

        time_feat = self.time_enc(dt)
        amt_feat = self.amount_enc(amt)

        x = torch.cat([node_feat, time_feat, amt_feat], dim=1)
        h = self.transform(x)

        return (weights.unsqueeze(1) * h).sum(0)

    def zebra_embed_ensemble(self, reservoirs, node, current_time):
        embs = []
        for key in ENSEMBLE:
            res = reservoirs[key].get(node)
            embs.append(self.zebra_embed_single(res, current_time))
        return torch.cat(embs, dim=0)


# =====================================================
# HELPERS
# =====================================================

def empty_topk(k):
    nodes = np.full(k, -1, dtype=np.int64)
    times = np.full(k, -1, dtype=np.int64)
    amts = np.zeros(k, dtype=np.float32)
    vals = np.zeros(k, dtype=np.float32)
    return nodes, times, amts, vals, 0


def scan_max_node_id(path, max_total_edges=None):
    max_id = 0
    seen = 0
    for src, dst, _, _ in iter_edges(path):
        if max_total_edges is not None:
            remaining = max_total_edges - seen
            if remaining <= 0:
                break
            src = src[:remaining]
            dst = dst[:remaining]
        if len(src) > 0:
            max_id = max(max_id, int(src.max()), int(dst.max()))
            seen += len(src)
    return max_id


# =====================================================
# MAIN
# =====================================================

def build_embeddings():
    sort_graph_if_needed()

    print("\nScanning max node id...")
    max_node_id = scan_max_node_id(GRAPH_PATH, MAX_TOTAL_EDGES)
    print("Max node id:", max_node_id)

    encoder = ZebraEncoder(max_node_id).to(DEVICE)
    encoder.eval()

    reservoirs = {key: {} for key in ENSEMBLE}
    m_vals = {key: defaultdict(float) for key in ENSEMBLE}

    # latest embedding per node
    saved_embeddings = {}
    saved_times = {}

    step = 0
    print("\nStreaming graph and building embeddings...")

    for block in iter_edges_grouped_by_time(GRAPH_PATH, max_total_edges=MAX_TOTAL_EDGES):
        if len(block) == 0:
            continue

        current_t = block[0][2]

        # Phase A: compute embeddings using history < current_t
        block_nodes = set()
        for u, v, _, _ in block:
            block_nodes.add(u)
            block_nodes.add(v)

        with torch.no_grad():
            for node in block_nodes:
                emb = encoder.zebra_embed_ensemble(reservoirs, node, current_t)
                saved_embeddings[node] = emb.cpu().numpy().astype(np.float32)
                saved_times[node] = current_t

        # Phase B: update reservoirs with current block
        for u, v, t, a in block:
            step += 1

            for alpha, beta in ENSEMBLE:
                key = (alpha, beta)

                if u in reservoirs[key]:
                    nodes_u, times_u, amts_u, vals_u, len_u = reservoirs[key][u]
                else:
                    nodes_u, times_u, amts_u, vals_u, len_u = empty_topk(TOP_K)

                if v in reservoirs[key]:
                    nodes_v, times_v, amts_v, vals_v, len_v = reservoirs[key][v]
                else:
                    nodes_v, times_v, amts_v, vals_v, len_v = empty_topk(TOP_K)

                mu = m_vals[key][u]
                mv = m_vals[key][v]

                new_u = update_reservoir_numba(
                    nodes_u, times_u, amts_u, vals_u, len_u,
                    nodes_v, times_v, amts_v, vals_v, len_v,
                    mu, alpha, beta, TOP_K,
                    v, t, a
                )

                new_v = update_reservoir_numba(
                    nodes_v, times_v, amts_v, vals_v, len_v,
                    nodes_u, times_u, amts_u, vals_u, len_u,
                    mv, alpha, beta, TOP_K,
                    u, t, a
                )

                reservoirs[key][u] = new_u
                reservoirs[key][v] = new_v

                m_vals[key][u] = update_m_numba(mu, beta)
                m_vals[key][v] = update_m_numba(mv, beta)

        if step % 100000 == 0:
            print(
                "edges:", f"{step:,}",
                "| current height:", current_t,
                "| saved embeddings:", len(saved_embeddings)
            )

    print("\nFinished.")
    print("Total edges streamed:", step)
    print("Total saved embeddings:", len(saved_embeddings))

    print("\nPreparing parquet output...")

    node_ids = np.array(list(saved_embeddings.keys()), dtype=np.int64)
    times = np.array([saved_times[n] for n in node_ids], dtype=np.int64)

    emb_matrix = np.vstack([saved_embeddings[n] for n in node_ids]).astype(np.float32)

    data = {
        "node_id": node_ids,
        "time": times,
    }

    for i in range(emb_matrix.shape[1]):
        data[f"emb_{i}"] = emb_matrix[:, i]

    df_out = pd.DataFrame(data)

    print("Output shape:", df_out.shape)
    print("Saving parquet to:", OUT_EMB_PATH)

    df_out.to_parquet(OUT_EMB_PATH, index=False)

    print("\nSaved embeddings to:", OUT_EMB_PATH)
    print(df_out.head())


if __name__ == "__main__":
    build_embeddings()