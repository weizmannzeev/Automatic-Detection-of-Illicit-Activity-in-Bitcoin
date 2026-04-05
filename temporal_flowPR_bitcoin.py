"""
temporal_ssl_stream_bitcoin.py
---------------------------------------

Temporal PageRank + label propagation
(Streaming version of Algorithm 1)

Based on:

Algorithm 1: stream processing
"""

import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

from data import load_data

from utils import (
    confusion_licit_illicit_table,
    save_ssl_experiment
)


# =====================================================
# PARAMETERS
# =====================================================

ALPHA = 0.85
BETA = 0.5

SEED_FRAC_ILL = 0.05
SEED_FRAC_LIC = 0.05


# =====================================================
# SEED SAMPLING
# =====================================================

def sample_seeds(nodes, frac):
    n = int(len(nodes) * frac)
    return set(random.sample(list(nodes), n))


# =====================================================
# STREAM TEMPORAL SSL
# =====================================================

def temporal_ssl_stream(df, illicit_set, licit_set, alpha=ALPHA, beta=BETA):

    # r(u) = accumulated class mass
    r = defaultdict(lambda: np.zeros(2))

    # s(u) = active walker mass
    s = defaultdict(lambda: np.zeros(2))

    edges = 0

    for row in df.itertuples(index=False):

        u = int(row.addr_id1)
        v = int(row.addr_id2)

        # ------------------------------------------------
        # TELEPORTATION STEP (Algorithm lines 3–4)
        # Seeds inject class mass into the system
        # ------------------------------------------------

        if u in licit_set:
            r[u][0] += (1 - alpha)
            s[u][0] += (1 - alpha)

        if u in illicit_set:
            r[u][1] += (1 - alpha)
            s[u][1] += (1 - alpha)

        # ------------------------------------------------
        # PROPAGATION STEP (Algorithm line 5)
        # ------------------------------------------------

        flow = s[u] * alpha

        r[v] += flow

        # ------------------------------------------------
        # WALKER UPDATE (Algorithm lines 6–11)
        # ------------------------------------------------

        if beta < 1:

            s[v] += flow * (1 - beta)
            s[u] *= beta

        else:

            s[v] += flow
            s[u] = 0

        edges += 1

        if edges % 1_000_000 == 0:
            print("processed edges:", edges)

    return r


# =====================================================
# NORMALIZATION
# =====================================================

def normalize_scores(r):

    total = np.zeros(2)

    for v in r.values():
        total += v

    for k in r:
        r[k] /= total

    return r


# =====================================================
# EVALUATION
# =====================================================

def evaluate(r, test_illicit, test_licit, df):

    nodes = list(test_illicit | test_licit)

    y_true = []
    y_pred = []

    for n in nodes:

        if n not in r:
            continue

        licit_mass, illicit_mass = r[n]

        pred = 1 if illicit_mass > licit_mass else 0

        if n in test_illicit:
            y_true.append(1)
        else:
            y_true.append(0)

        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["licit", "illicit"],
            digits=4
        )
    )

    table = confusion_licit_illicit_table(y_true, y_pred)

    print("\nConfusion matrix\n")
    print(table)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    precision_illicit = tp / (tp + fp) if tp + fp > 0 else 0
    recall_illicit = tp / (tp + fn) if tp + fn > 0 else 0

    f1_illicit = (
        2 * precision_illicit * recall_illicit /
        (precision_illicit + recall_illicit)
        if precision_illicit + recall_illicit > 0 else 0
    )

    precision_licit = tn / (tn + fn) if tn + fn > 0 else 0
    recall_licit = tn / (tn + fp) if tn + fp > 0 else 0

    f1_licit = (
        2 * precision_licit * recall_licit /
        (precision_licit + recall_licit)
        if precision_licit + recall_licit > 0 else 0
    )

    save_ssl_experiment(
        outfile="temporal_flowpr_ssl.xlsx",
        iteration=1,
        seed_method="temporal_flowpr_ssl",
        mu=0,
        sigma=0,
        alpha=ALPHA,

        n_nodes=len(set(df.addr_id1) | set(df.addr_id2)),
        n_edges=len(df),

        seed_ill=test_illicit,
        seed_lic=test_licit,

        eval_size=len(y_true),

        precision_illicit=precision_illicit,
        recall_illicit=recall_illicit,
        f1_illicit=f1_illicit,

        precision_licit=precision_licit,
        recall_licit=recall_licit,
        f1_licit=f1_licit,

        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,

        graph_sorted=True
    )


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    print("Loading graph...")

    df, labels, illicit_set, licit_set, known_ids = load_data(use_temporal=True)

    print("Edges:", len(df))
    print("Known nodes:", len(known_ids))

    # ------------------------------------------------
    # Create seed / test split
    # ------------------------------------------------

    seed_illicit = sample_seeds(illicit_set, SEED_FRAC_ILL)
    seed_licit = sample_seeds(licit_set, SEED_FRAC_LIC)

    test_illicit = illicit_set - seed_illicit
    test_licit = licit_set - seed_licit

    print("\nSeed nodes")
    print("Seed illicit:", len(seed_illicit))
    print("Seed licit:", len(seed_licit))

    print("\nTest nodes")
    print("Test illicit:", len(test_illicit))
    print("Test licit:", len(test_licit))

    print("\nRunning temporal SSL streaming...")

    r = temporal_ssl_stream(df, seed_illicit, seed_licit)

    print("\nNormalizing scores...")

    r = normalize_scores(r)

    print("\nEvaluating...")

    evaluate(r, test_illicit, test_licit, df)