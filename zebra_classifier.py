import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =====================================================
# PATHS
# =====================================================

EMB_PATH = "zebra_node_embeddings.parquet"

LABEL_PATH = os.path.expanduser(
    "~/.cache/RP/cleaned_seed_addresses_with_ids_filtered.csv"
)

OUTPUT_FILE = "zebra_classifier_results.txt"


# =====================================================
# SETTINGS
# =====================================================

TARGET_TRAIN_LABELED = 20000
SEED = 42

np.random.seed(SEED)


# =====================================================
# LOAD LABELS
# =====================================================

def load_labels():

    df = pd.read_csv(LABEL_PATH)

    df = df.dropna(subset=["address_id", "risk_level"])

    df["address_id"] = df["address_id"].astype(np.int64)

    df["label"] = (df["risk_level"] == "illicit").astype(int)

    return df[["address_id", "label"]].drop_duplicates()


# =====================================================
# TEMPORAL SPLIT
# =====================================================

def build_streaming_disjoint_split(df, target_train_labeled):

    df = df.sort_values(["time", "node_id"]).reset_index(drop=True)

    unique_nodes = df["node_id"].drop_duplicates().tolist()

    train_nodes = set(unique_nodes[:target_train_labeled])

    test_nodes = set(unique_nodes[target_train_labeled:])

    return train_nodes, test_nodes


# =====================================================
# METRICS
# =====================================================

def print_and_save_metrics(y_true, y_pred, y_score):

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    lines = []

    lines.append("\n================ FINAL METRICS ================")
    lines.append("Class      Precision  Recall     F1-score   Support")

    lines.append(
        f"Licit      {precision[0]:.4f}    {recall[0]:.4f}    {f1[0]:.4f}    {support[0]}"
    )

    lines.append(
        f"Illicit    {precision[1]:.4f}    {recall[1]:.4f}    {f1[1]:.4f}    {support[1]}"
    )

    lines.append(f"\nAccuracy: {round(acc,6)}")
    lines.append(f"ROC-AUC : {round(auc,6) if auc == auc else 'nan'}")

    lines.append("\nConfusion matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Target licit", "Target illicit"],
        columns=["Pred licit", "Pred illicit"],
    )

    lines.append(cm_df.to_string())

    text = "\n".join(lines)

    print(text)

    with open(OUTPUT_FILE, "w") as f:
        f.write(text)


# =====================================================
# MAIN
# =====================================================

def main():

    print("\nLoading embeddings...")

    emb = pd.read_parquet(EMB_PATH)

    print("Embeddings shape:", emb.shape)

    labels = load_labels()

    print("Labels:", labels.shape)

    df = emb.merge(
        labels,
        left_on="node_id",
        right_on="address_id",
    )

    print("After merge:", df.shape)
    #  # =====================================================
    # # REMOVE ZERO EMBEDDINGS (ADDED)
    # # =====================================================

    # feature_cols = [c for c in df.columns if c.startswith("emb_")]

    # mask_nonzero = (df[feature_cols].abs().sum(axis=1) > 0)

    # print("\nBefore filtering:", df.shape)

    # df = df[mask_nonzero].reset_index(drop=True)

    # print("After removing zero embeddings:", df.shape)
    # print("Removed fraction:", 1 - mask_nonzero.mean())


    # =====================================================
    # EMBEDDING DIAGNOSTICS
    # =====================================================

    feature_cols = [c for c in df.columns if c.startswith("emb_")]

    print("\nEmbedding variance (mean):")
    print(df[feature_cols].var().mean())

    zero_fraction = (df[feature_cols].abs().sum(axis=1) == 0).mean()

    print("Fraction of zero embeddings:", zero_fraction)

    print("\nEmbedding statistics:")
    print(df[feature_cols].describe())


    # =====================================================
    # TEMPORAL SPLIT
    # =====================================================

    train_nodes, test_nodes = build_streaming_disjoint_split(
        df,
        TARGET_TRAIN_LABELED,
    )

    print("\nTrain nodes:", len(train_nodes))
    print("Test nodes :", len(test_nodes))


    # =====================================================
    # DATASETS
    # =====================================================

    X_train = df[df.node_id.isin(train_nodes)][feature_cols].values
    y_train = df[df.node_id.isin(train_nodes)]["label"].values

    X_test = df[df.node_id.isin(test_nodes)][feature_cols].values
    y_test = df[df.node_id.isin(test_nodes)]["label"].values

    print("\nTrain samples:", X_train.shape[0])
    print("Test samples :", X_test.shape[0])

    print("\nTrain class distribution")
    print(pd.Series(y_train).value_counts())

    print("\nTest class distribution")
    print(pd.Series(y_test).value_counts())


    # =====================================================
    # SCALING
    # =====================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    print("\nTraining Logistic Regression...")

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)


    # =====================================================
    # PREDICT
    # =====================================================

    print("\nPredicting...")

    probs = clf.predict_proba(X_test)[:, 1]


    # =====================================================
    # THRESHOLD SEARCH
    # =====================================================

    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    best_threshold = 0.5

    for p, r, t in zip(precision, recall, thresholds):

        if p >= 0.8:
            best_threshold = t
            break

    print("\nChosen threshold:", best_threshold)

    preds = (probs > best_threshold).astype(int)


    # =====================================================
    # METRICS
    # =====================================================

    print_and_save_metrics(y_test, preds, probs)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()