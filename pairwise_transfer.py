"""
Pairwise transfer learning with alignment + class balancing
"""

# ===============================
# IMPORTS
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_graph_and_labels, build_pyg_graphs_from_subgraphs
from egonets import extract_k_hop_illicit_clusters, compute_triangle_stats
from alignment import compute_penalty

#
from model import (
    subgraph_to_tensor_ppgn,
    NodePPGN_3WL,
    NodePairGNN,
    NodeGCN
)

# ===============================
# CONFIG
# ===============================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_HOP = 2
HIDDEN = 16
LR = 0.01
EPOCHS = 80
THRESHOLD = 0.5
OT_BETA = 0.1
ALIGN_WEIGHT = 0.5

PAIRS = [

    # ===== BASE (medium ↔ medium) =====
    (6, 12), (12, 6),
    (6, 11), (11, 6),
    (6, 19), (19, 6),
    (11, 19), (19, 11),
    (24, 6), (6, 24),
    (25, 11), (11, 25),

    # ===== BORDERLINE CASES =====
    (10, 6), (6, 10),
    (16, 11), (11, 16),
    (14, 6), (6, 14),
    (8, 12), (12, 8),

    # ===== LARGE-ish (scaling check) =====
    (32, 6), (6, 32),
    (32, 11), (11, 32),

    # ===== SMALL (sanity check) =====
    (7, 6), (6, 7),
    (15, 6), (6, 15),
    (22, 7),
    (20, 7),

]
MODELS = ["ppgn_3wl", "pairgnn", "gcn"]

# ===============================
# SEED
# ===============================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ===============================
# LOAD DATA
# ===============================
print("Loading data...")
df, merged, illicit_set, licit_set, known_ids = load_graph_and_labels()

df = df[
    df["addr_id1"].isin(known_ids) &
    df["addr_id2"].isin(known_ids)
]

# ===============================
# SUBGRAPHS
# ===============================
print("Extracting subgraphs...")
subgraphs, sizes = extract_k_hop_illicit_clusters(df, illicit_set, k=K_HOP)


# ===============================
# GCN FEATURES (structure only)
# ===============================

features = {nid: torch.zeros(1, dtype=torch.float32) for nid in known_ids}

pyg_graphs = build_pyg_graphs_from_subgraphs(
    subgraphs,
    illicit_set,
    address_embeddings=features,
)


# ===============================
# UTILS
# ===============================
def compute_pos_weight(y):
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()
    return torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)



#
def eval_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["non-illicit", "illicit"],
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return report, tp, fp, fn, tn

# ===============================
# RUN
# ===============================
def run(model_name, tr, te):
    print(f"\n=== {model_name} {tr}->{te} ===")

    n_tri, tri_density = compute_triangle_stats(subgraphs[te])

    if model_name in ["ppgn_3wl", "pairgnn"]:
        X_A, Y_A, mask_A = subgraph_to_tensor_ppgn(subgraphs[tr], illicit_set, DEVICE)
        X_B, Y_B, mask_B = subgraph_to_tensor_ppgn(subgraphs[te], illicit_set, DEVICE)

        in_channels = X_A.shape[-1]

        model = (
            NodePPGN_3WL(in_channels, HIDDEN)
            if model_name == "ppgn_3wl"
            else NodePairGNN(in_channels, HIDDEN)
        ).to(DEVICE)

        pos_weight = compute_pos_weight(Y_A)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()

            logits = model(X_A)
            loss_cls = loss_fn(logits, Y_A)
            z_A = model.get_embeddings(X_A)
            with torch.no_grad():
                z_B = model.get_embeddings(X_B)
            z_A = F.normalize(z_A, dim=1)
            z_B = F.normalize(z_B, dim=1)
            loss_align, K = compute_penalty(z_A, z_B, beta=OT_BETA)
            loss = loss_cls + ALIGN_WEIGHT * loss_align

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 20 == 0:
                print(f"epoch {epoch} "
        f"cls {loss_cls.item():.4f} "
        f"align {loss_align.item():.4f} "
        f"total {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_B))
            preds = (probs > THRESHOLD).float()

        print("prob mean/min/max:",
              round(probs.mean().item(), 4),
              round(probs.min().item(), 4),
              round(probs.max().item(), 4))

        y_true = Y_B.cpu().numpy()
        y_pred = preds.cpu().numpy()

    else:
        G_A = pyg_graphs[tr].to(DEVICE)
        G_B = pyg_graphs[te].to(DEVICE)

        model = NodeGCN(G_A.x.shape[1], HIDDEN).to(DEVICE)

        pos_weight = compute_pos_weight(G_A.y)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train()

            logits = model(G_A.x, G_A.edge_index)
            loss_cls = loss_fn(logits, G_A.y.float())

            z_A = model.get_embeddings(G_A.x, G_A.edge_index)
            with torch.no_grad():
                z_B = model.get_embeddings(G_B.x, G_B.edge_index)
            z_A = F.normalize(z_A, dim=1)
            z_B = F.normalize(z_B, dim=1)

            loss_align, K = compute_penalty(z_A, z_B, beta=OT_BETA)
            loss = loss_cls + ALIGN_WEIGHT * loss_align

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 20 == 0:
                print(
                    f"epoch {epoch} "
                    f"cls {loss_cls.item():.4f} "
                    f"align {loss_align.item():.4f} "
                    f"total {loss.item():.4f}"
                )
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(G_B.x, G_B.edge_index))
            preds = (probs > THRESHOLD).float()

        print("prob mean/min/max:",
              round(probs.mean().item(), 4),
              round(probs.min().item(), 4),
              round(probs.max().item(), 4))

        y_true = G_B.y.cpu().numpy()
        y_pred = preds.cpu().numpy()

    print("predicted positives:", int(y_pred.sum()), "/", int(y_true.sum()))

    report, tp, fp, fn, tn = eval_metrics(y_true, y_pred)

    return {
        "model": model_name,
        "train": tr,
        "test": te,
        "f1": report["illicit"]["f1-score"],
        "precision": report["illicit"]["precision"],
        "recall": report["illicit"]["recall"],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "triangles": n_tri,
        "density": tri_density,
        "size_train": len(subgraphs[tr]),
        "size_test": len(subgraphs[te]),
    }

# ===============================
# RUN ALL
# ===============================
results = []

for tr, te in PAIRS:
    for m in MODELS:
        try:
            results.append(run(m, tr, te))
        except Exception as e:
            print(f"[ERROR] {m} {tr}->{te}: {e}")

df_res = pd.DataFrame(results)

print("\nRESULTS:")
print(df_res)

df_res.to_excel("pairwise_transfer_results.xlsx", index=False)
print("saved")