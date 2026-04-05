from sklearn.metrics import confusion_matrix
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import glob
import os


# =========================================================
# Confusion matrix table
# =========================================================
def confusion_licit_illicit_table(y_true, y_pred, mask=None):
    """
    Return confusion matrix table for licit (0) vs illicit (1).
    """

    if mask is not None:
        yt = y_true[mask]
        yp = y_pred[mask]
    else:
        yt = y_true
        yp = y_pred

    cm = confusion_matrix(yt, yp, labels=[0, 1])

    table = pd.DataFrame(
        cm,
        index=["True licit", "True illicit"],
        columns=["Pred licit", "Pred illicit"]
    )

    return table


# =========================================================
# Check if parquet is sorted
# =========================================================
def check_sorted_parquet(path, column="height"):
    """
    Check if parquet file is sorted by a given column.
    """

    pf = pq.ParquetFile(path)

    last = None

    for batch in pf.iter_batches(batch_size=1_000_000, columns=[column]):

        values = np.array(batch[column])

        if last is not None and values[0] < last:
            return False

        if np.any(values[1:] < values[:-1]):
            return False

        last = values[-1]

    return True


# =========================================================
# Save experiment results
# =========================================================
def save_ssl_experiment(
    outfile,
    iteration,
    seed_method,
    mu,
    sigma,
    alpha,
    n_nodes,
    n_edges,
    seed_ill,
    seed_lic,
    eval_size,
    precision_illicit,
    recall_illicit,
    f1_illicit,
    precision_licit,
    recall_licit,
    f1_licit,
    tp,
    fp,
    fn,
    tn,
    graph_sorted
):
    """
    Save experiment parameters + metrics.
    """

    results = {
        "iteration": iteration,
        "seed_method": seed_method,
        "mu": mu,
        "sigma": sigma,
        "alpha": alpha,

        "n_nodes": n_nodes,
        "n_edges": n_edges,

        "n_seeds_illicit": len(seed_ill),
        "n_seeds_licit": len(seed_lic),

        "graph_sorted": graph_sorted,

        "eval_nodes": eval_size,

        "precision_illicit": precision_illicit,
        "recall_illicit": recall_illicit,
        "f1_illicit": f1_illicit,

        "precision_licit": precision_licit,
        "recall_licit": recall_licit,
        "f1_licit": f1_licit,

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }

    df = pd.DataFrame([results])

    df.to_excel(outfile, index=False)

    print(f"[Saved experiment → {outfile}]")





def print_confusion_matrices_from_files(folder="."):

    files = glob.glob(os.path.join(folder, "*.xlsx"))

    for file in sorted(files):

        name = os.path.basename(file)

        if name.startswith("~$"):
            continue

        df = pd.read_excel(file)

        row = df.iloc[-1]

        tp = int(row["TP"])
        fp = int(row["FP"])
        fn = int(row["FN"])
        tn = int(row["TN"])

        precision_illicit = row["precision_illicit"]
        recall_illicit = row["recall_illicit"]

        support_licit = tn + fp
        support_illicit = fn + tp

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        method = "_".join(name.split("_")[-2:])

        print("\n========================================")
        print("Method:", method)

        print("\nConfusion matrix (with support)")
        print("                 Pred licit   Pred illicit   Support")

        print(f"True licit     {tn:12d} {fp:14d} {support_licit:10d}")
        print(f"True illicit   {fn:12d} {tp:14d} {support_illicit:10d}")

        print("\nMetrics")
        print("Accuracy:", round(accuracy, 4))
        print("Precision (illicit):", round(precision_illicit, 4))
        print("Recall (illicit):", round(recall_illicit, 4))

print_confusion_matrices_from_files()

def balance_classes(y_true, y_pred, mask=None, random_state=42):
    """
    Downsample majority class to match minority class.
    Returns balanced y_true, y_pred.
    """

    if mask is not None:
        yt = y_true[mask]
        yp = y_pred[mask]
    else:
        yt = y_true
        yp = y_pred

    # indices
    idx_licit = np.where(yt == 0)[0]
    idx_illicit = np.where(yt == 1)[0]

    n = min(len(idx_licit), len(idx_illicit))

    rng = np.random.default_rng(random_state)

    idx_licit_sample = rng.choice(idx_licit, n, replace=False)
    idx_illicit_sample = rng.choice(idx_illicit, n, replace=False)

    idx_balanced = np.concatenate([idx_licit_sample, idx_illicit_sample])

    return yt[idx_balanced], yp[idx_balanced]