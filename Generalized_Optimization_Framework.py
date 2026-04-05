"""
Konstantin_iterative_numba_full_evalU_directed.py
--------------------------------------------------------
Numba-accelerated iterative implementation of the
Generalized Optimization Framework for
Graph-based Semi-Supervised Learning (DIRECTED)
(Avrachenkov et al., SIAM 2011)

F_{t+1} = (1 - α) * Y + α * D_in^{-σ} W D_out^{σ-1} F_t
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.metrics import classification_report, confusion_matrix

from data import load_data

from utils import (
    confusion_licit_illicit_table,
    save_ssl_experiment
)

from seed_selection import (
    compute_pagerank,
    compute_degree,
    select_seeds,
    select_seeds_random,
    compute_personalized_pagerank,
    compute_temporal_flow_pagerank,
)


# ======================================================
# Hyperparameters
# ======================================================

MU = 0.5
SIGMA = 0
ALPHA = 2 / (2 + MU)

N_ITER = 20
SAVE_EVERY = 5

OUTFILE = "generalized_ssl_iterative_numba_directed_evalU.xlsx"

SEED_METHOD = "flowpr"  # random | pagerank | degree | ppr | flowpr
SEED_RANDOM_STATE = 42

SEED_FRAC_ILL = 0.05
SEED_FRAC_LIC = 0.05


# ======================================================
# Load graph
# ======================================================

print("[Data] Loading static graph for SSL...")
df, merged, illicit_set, licit_set, known_ids = load_data(use_temporal=False)
print("[Data] Loading temporal graph for flowPR...")
df_temporal, _, _, _, _ = load_data(use_temporal=True)

print(f"[Data] {len(df)} edges, {len(known_ids)} known addresses.")


edges = df[["addr_id1", "addr_id2"]].to_numpy(np.int64)

nodes_unique = np.unique(edges)

node_to_idx = {n: i for i, n in enumerate(nodes_unique)}
idx_to_node = np.array(nodes_unique, dtype=np.int64)

N = len(nodes_unique)

print(f"[Graph] {N} nodes, {len(edges)} directed edges.")


src = np.array([node_to_idx[a] for a in edges[:, 0]], dtype=np.int64)
dst = np.array([node_to_idx[b] for b in edges[:, 1]], dtype=np.int64)


# ======================================================
# Connected components analysis
# ======================================================

import networkx as nx

print("[Graph] Computing connected components...")

G = nx.DiGraph()
G.add_edges_from(zip(src, dst))

components = list(nx.weakly_connected_components(G))

print(f"[Graph] number of components: {len(components)}")
print(f"[Graph] largest component size: {max(len(c) for c in components)}")



# ======================================================
# Directed degrees
# ======================================================

deg_out = np.zeros(N)
deg_in = np.zeros(N)

for k in range(len(src)):
    deg_out[src[k]] += 1
    deg_in[dst[k]] += 1


# ======================================================
# True labels
# ======================================================

print("[Init] Setting labels...")

y_true = -np.ones(N, dtype=np.int32)

illicit_idx = np.array(
    [node_to_idx[n] for n in illicit_set if n in node_to_idx],
    dtype=np.int64
)

licit_idx = np.array(
    [node_to_idx[n] for n in licit_set if n in node_to_idx],
    dtype=np.int64
)

y_true[illicit_idx] = 1
y_true[licit_idx] = 0

known_mask = y_true != -1


print("[Labels] Encoding:")
print("  -1 -> unknown")
print("   0 -> licit")
print("   1 -> illicit")

unique_labels, counts = np.unique(y_true, return_counts=True)

print("[Labels] Distribution in y_true:")
for lbl, cnt in zip(unique_labels, counts):
    if lbl == -1:
        name = "unknown"
    elif lbl == 0:
        name = "licit"
    elif lbl == 1:
        name = "illicit"
    else:
        name = "other"
    print(f"  label {lbl} ({name}): {cnt}")

print(f"[Labels] Known licit nodes: {len(licit_idx)}")
print(f"[Labels] Known illicit nodes: {len(illicit_idx)}")
print(f"[Labels] Total known nodes: {known_mask.sum()}")
print(f"[Labels] Total unknown nodes: {(~known_mask).sum()}")

# ======================================================
# Seed selection
# ======================================================

print(f"[Seeds] Method: {SEED_METHOD}")

if SEED_METHOD == "random":

    seed_ill, seed_lic = select_seeds_random(
        illicit_idx,
        licit_idx,
        SEED_FRAC_ILL,
        SEED_FRAC_LIC,
        seed=SEED_RANDOM_STATE
    )

elif SEED_METHOD == "pagerank":

    scores = compute_pagerank(src, dst, N)

    seed_ill, seed_lic = select_seeds(
        scores,
        illicit_idx,
        licit_idx,
        SEED_FRAC_ILL,
        SEED_FRAC_LIC
    )

elif SEED_METHOD == "degree":

    scores = compute_degree(src, dst, N)

    seed_ill, seed_lic = select_seeds(
        scores,
        illicit_idx,
        licit_idx,
        SEED_FRAC_ILL,
        SEED_FRAC_LIC
    )
elif SEED_METHOD == "ppr":

    # personalized pagerank from illicit nodes
    scores_ill = compute_personalized_pagerank(
        src, dst, N, teleport_idx=illicit_idx
    )

    # personalized pagerank from licit nodes
    scores_lic = compute_personalized_pagerank(
        src, dst, N, teleport_idx=licit_idx
    )

    n_ill = max(1, int(len(illicit_idx) * SEED_FRAC_ILL))
    n_lic = max(1, int(len(licit_idx) * SEED_FRAC_LIC))

    seed_ill = illicit_idx[
        np.argsort(-scores_ill[illicit_idx])[:n_ill]
    ]

    seed_lic = licit_idx[
        np.argsort(-scores_lic[licit_idx])[:n_lic]
    ]    

elif SEED_METHOD == "flowpr":

    print("[flowPR] Computing temporal flow PageRank...")

    edges_temp = df_temporal[["addr_id1", "addr_id2"]].to_numpy(np.int64)

    src_temp = np.array([node_to_idx[a] for a in edges_temp[:, 0]], dtype=np.int64)
    dst_temp = np.array([node_to_idx[b] for b in edges_temp[:, 1]], dtype=np.int64)

    scores = compute_temporal_flow_pagerank(
        src_temp,
        dst_temp,
        N,
        alpha=0.85,
        beta=0.0,
        gamma=1.0
    )
    

    seed_ill, seed_lic = select_seeds(
        scores,
        illicit_idx,
        licit_idx,
        SEED_FRAC_ILL,
        SEED_FRAC_LIC
    )

    print("[flowPR] score min:", scores.min())
    print("[flowPR] score mean:", scores.mean())
    print("[flowPR] score max:", scores.max())
    print("[flowPR] first illicit seeds:", seed_ill[:10])
    print("[flowPR] first licit seeds:", seed_lic[:10])

else:

    raise ValueError("Unknown seed method")


L_mask = np.zeros(N, dtype=np.bool_)

L_mask[seed_ill] = True
L_mask[seed_lic] = True

U_mask = ~L_mask

print(
    f"[Seeds] illicit={len(seed_ill)} "
    f"licit={len(seed_lic)} "
    f"unlabeled={U_mask.sum()}"

)
print("[Seeds] Details:")
print(f"  total seed nodes (L): {L_mask.sum()}")
print(f"  total unlabeled nodes (U): {U_mask.sum()}")
print(f"  illicit seeds: {len(seed_ill)} / {len(illicit_idx)}")
print(f"  licit seeds: {len(seed_lic)} / {len(licit_idx)}")


# ======================================================
# Initial label matrix
# ======================================================

Y = np.zeros((N, 2))

Y[seed_lic, 0] = 1.0 / len(seed_lic)
Y[seed_ill, 1] = 1.0 / len(seed_ill)

F = Y.copy()

print(f"[Init] α={ALPHA:.3f}, σ={SIGMA}, μ={MU}")


# ======================================================
# Propagation step
# ======================================================

@njit(parallel=True, fastmath=True)
def propagate_step_directed(src, dst, deg_out, deg_in, F, Y, alpha, sigma, N):

    F_new = np.empty_like(F)

    s = np.zeros((N, 2))

    for k in prange(len(src)):

        i = src[k]
        j = dst[k]

        doi = max(deg_out[i], 1)

        s[j, 0] += F[i, 0] / (doi ** (1 - sigma))
        s[j, 1] += F[i, 1] / (doi ** (1 - sigma))

    for j in prange(N):

        dij = max(deg_in[j], 1)

        F_new[j, 0] = (1 - alpha) * Y[j, 0] + alpha * (s[j, 0] / (dij ** sigma))
        F_new[j, 1] = (1 - alpha) * Y[j, 1] + alpha * (s[j, 1] / (dij ** sigma))

    return F_new


# ======================================================
# SSL iterations
# ======================================================

print("[Run] SSL propagation...")

for t in range(1, N_ITER + 1):

    F = propagate_step_directed(
        src,
        dst,
        deg_out,
        deg_in,
        F,
        Y,
        ALPHA,
        SIGMA,
        N
    )

    if t % SAVE_EVERY == 0 or t == N_ITER:

        print(f"[Iteration {t}]")

        y_pred = (F[:, 1] > F[:, 0]).astype(np.int32)

        eval_mask = U_mask & known_mask
        print("[Eval] Subset statistics:")
        print(f"  eval size: {eval_mask.sum()}")
        print(f"  true licit in eval: {(y_true[eval_mask] == 0).sum()}")
        print(f"  true illicit in eval: {(y_true[eval_mask] == 1).sum()}")
        print(f"  predicted licit in eval: {(y_pred[eval_mask] == 0).sum()}")
        print(f"  predicted illicit in eval: {(y_pred[eval_mask] == 1).sum()}")

        yt = y_true[eval_mask]
        yp = y_pred[eval_mask]

        rep = classification_report(
            yt,
            yp,
            target_names=["licit", "illicit"],
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(yt, yp, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        save_ssl_experiment(
            outfile=OUTFILE,
            iteration=t,
            seed_method=SEED_METHOD,
            mu=MU,
            sigma=SIGMA,
            alpha=ALPHA,
            n_nodes=N,
            n_edges=len(src),
            seed_ill=seed_ill,
            seed_lic=seed_lic,
            eval_size=eval_mask.sum(),
            precision_illicit=rep["illicit"]["precision"],
            recall_illicit=rep["illicit"]["recall"],
            f1_illicit=rep["illicit"]["f1-score"],
            precision_licit=rep["licit"]["precision"],
            recall_licit=rep["licit"]["recall"],
            f1_licit=rep["licit"]["f1-score"],
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            graph_sorted=(SEED_METHOD == "flowpr")
        )


# ======================================================
# Final confusion matrix
# ======================================================

print("\nFinal confusion matrix")

cm_table = confusion_licit_illicit_table(
    y_true,
    y_pred,
    mask=eval_mask
)

print(cm_table)

