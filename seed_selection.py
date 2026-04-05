import numpy as np


# =========================================================
# PageRank
# =========================================================
def compute_pagerank(src, dst, N, alpha=0.85, n_iter=20):
    """
    Compute PageRank scores.

    Parameters
    ----------
    src : array
        source nodes
    dst : array
        destination nodes
    N : int
        number of nodes
    alpha : float
        damping factor
    n_iter : int
        number of iterations

    Returns
    -------
    r : array
        pagerank score for each node
    """

    # out-degree
    deg_out = np.zeros(N)

    for i in src:
        deg_out[i] += 1

    # initialize pagerank
    r = np.ones(N) / N

    # iterative update
    for _ in range(n_iter):

        new_r = np.zeros(N)

        for k in range(len(src)):

            i = src[k]
            j = dst[k]

            if deg_out[i] > 0:
                new_r[j] += r[i] / deg_out[i]

        r = alpha * new_r + (1 - alpha) / N

    return r

# =========================================================
# Personalized PageRank
# =========================================================
def compute_personalized_pagerank(src, dst, N, teleport_idx, alpha=0.85, n_iter=20):
    """
    Compute Personalized PageRank scores.

    Parameters
    ----------
    src : array
        source nodes
    dst : array
        destination nodes
    N : int
        number of nodes
    teleport_idx : array
        nodes where teleportation happens
    alpha : float
        damping factor
    n_iter : int
        number of iterations

    Returns
    -------
    r : array
        pagerank score for each node
    """

    # out-degree
    deg_out = np.zeros(N)

    for i in src:
        deg_out[i] += 1

    # teleportation vector
    v = np.zeros(N)
    v[teleport_idx] = 1.0 / len(teleport_idx)

    # initialize
    r = v.copy()

    for _ in range(n_iter):

        new_r = np.zeros(N)

        for k in range(len(src)):

            i = src[k]
            j = dst[k]

            if deg_out[i] > 0:
                new_r[j] += r[i] / deg_out[i]

        r = alpha * new_r + (1 - alpha) * v

    return r

# =========================================================
# Degree
# =========================================================
def compute_degree(src, dst, N):
    """
    Compute node degree (in + out).
    """

    deg = np.zeros(N)

    for i in src:
        deg[i] += 1

    for j in dst:
        deg[j] += 1

    return deg


# =========================================================
# Seed selection using scores
# =========================================================
def select_seeds(scores, illicit_idx, licit_idx, frac_ill, frac_lic):
    """
    Select seeds based on node scores (pagerank, degree, etc.)
    """

    n_ill = max(1, int(len(illicit_idx) * frac_ill))
    n_lic = max(1, int(len(licit_idx) * frac_lic))

    illicit_scores = scores[illicit_idx]
    licit_scores = scores[licit_idx]

    seed_ill = illicit_idx[np.argsort(-illicit_scores)[:n_ill]]
    seed_lic = licit_idx[np.argsort(-licit_scores)[:n_lic]]

    return seed_ill, seed_lic


# =========================================================
# Random seed selection
# =========================================================
def select_seeds_random(illicit_idx, licit_idx, frac_ill, frac_lic, seed=42):
    """
    Random seed selection.
    """

    rng = np.random.default_rng(seed)

    n_ill = max(1, int(len(illicit_idx) * frac_ill))
    n_lic = max(1, int(len(licit_idx) * frac_lic))

    seed_ill = rng.choice(illicit_idx, size=n_ill, replace=False)
    seed_lic = rng.choice(licit_idx, size=n_lic, replace=False)

    return seed_ill, seed_lic

def compute_temporal_flow_pagerank(src, dst, N, alpha=0.85, beta=0.5, gamma=1.0):
    """
    Streaming temporal flowPR score.

    Parameters
    ----------
    src : array
        source nodes (already reindexed: 0..N-1)
    dst : array
        destination nodes (already reindexed: 0..N-1)
    N : int
        number of nodes
    alpha : float
        damping / continuation probability
    beta : float
        retained flow fraction
    gamma : float
        decay on accumulated rank

    Returns
    -------
    scores : array of shape (N,)
        normalized temporal flowPR scores
    """

    RS = np.zeros(N, dtype=np.float64)
    current = np.zeros(N, dtype=np.float64)

    for k in range(len(src)):
        u = src[k]
        v = dst[k]

        teleport = 1.0 - alpha

        RS[u] = RS[u] * gamma + teleport
        RS[v] = RS[v] * gamma + (current[u] + teleport) * alpha

        current[v] += (current[u] + teleport) * alpha * (1.0 - beta)
        current[u] *= beta

    total = RS.sum()
    if total > 0:
        RS /= total

    return RS