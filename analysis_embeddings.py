import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def pairwise_diff_vectors(X):

    n = X.shape[0]

    diffs = []

    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(X[i] - X[j])

    return np.asarray(diffs)


def relational_alignment_score(X, Y):

    Dx = pairwise_diff_vectors(X)
    Dy = pairwise_diff_vectors(Y)

    Dx = Dx / (np.linalg.norm(Dx, axis=1, keepdims=True) + 1e-8)
    Dy = Dy / (np.linalg.norm(Dy, axis=1, keepdims=True) + 1e-8)

    sims = np.sum(Dx * Dy, axis=1)

    return sims.mean()


def patient_matching_accuracy(X, Y):

    D = cdist(X, Y, metric="cosine")

    nn = np.argmin(D, axis=1)

    correct = np.arange(len(X))

    return np.mean(nn == correct)


def distance_matrix_correlation(X, Y):

    DX = squareform(
        pdist(X, metric="cosine")
    )

    DY = squareform(
        pdist(Y, metric="cosine")
    )

    corr, _ = spearmanr(
        DX.ravel(),
        DY.ravel()
    )

    return corr


def subtype_separation_ratio(X, labels):

    D = cdist(X, X, metric="cosine")

    within = []
    between = []

    n = len(labels)

    for i in range(n):
        for j in range(i+1, n):

            if labels[i] == labels[j]:
                within.append(D[i,j])
            else:
                between.append(D[i,j])

    return np.mean(between) / np.mean(within)



def neighbor_agreement(X, Y, k=10):

    nn_x = NearestNeighbors(
        n_neighbors=k+1,
        metric='cosine'
    ).fit(X)

    nn_y = NearestNeighbors(
        n_neighbors=k+1,
        metric='cosine'
    ).fit(Y)

    idx_x = nn_x.kneighbors(
        return_distance=False
    )[:,1:]

    idx_y = nn_y.kneighbors(
        return_distance=False
    )[:,1:]

    overlaps = []

    for a,b in zip(idx_x, idx_y):
        overlaps.append(
            len(set(a) & set(b))/k
        )

    return np.mean(overlaps)



def compute_silhouette(X, labels):

    return silhouette_score(
        X,
        labels,
        metric="cosine"
    )





def save_umap(X, labels, title, filename):

    reducer = umap.UMAP(
        metric='cosine',
        random_state=42
    )

    z = reducer.fit_transform(X)

    plt.figure(figsize=(8,6))

    for cls in np.unique(labels):
        idx = labels == cls

        plt.scatter(
            z[idx,0],
            z[idx,1],
            label=str(cls),
            alpha=0.8
        )

    plt.legend()
    plt.title(title)

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()




def knn_subtype_purity(X, labels, k=10):

    nn = NearestNeighbors(
        n_neighbors=k+1,
        metric='cosine'
    ).fit(X)

    idx = nn.kneighbors(
        return_distance=False
    )[:,1:]

    purities = []

    for i, neigh in enumerate(idx):

        purities.append(
            np.mean(labels[neigh] == labels[i])
        )

    return np.mean(purities)

def recall_at_k(X, Y, k=5):

    D = cdist(X, Y, metric="cosine")

    nearest = np.argsort(D, axis=1)[:, :k]

    hits = []

    for i in range(len(X)):
        hits.append(i in nearest[i])

    return np.mean(hits)


def evaluate_cross_modal_metrics(
    cna_np,
    met_np,
    mrna_np,
    rppa_np
):

    views = {
        "cna": cna_np,
        "met": met_np,
        "mrna": mrna_np,
        "rppa": rppa_np
    }

    keys = list(views.keys())

    alignment_scores = []
    matching_scores = []
    recall5_scores = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):

            X = views[keys[i]]
            Y = views[keys[j]]

            alignment_scores.append(
                relational_alignment_score(X, Y)
            )

            matching_scores.append(
                patient_matching_accuracy(X, Y)
            )

            recall5_scores.append(
                recall_at_k(X, Y, k=5)
            )

    return {
        "alignment": np.mean(alignment_scores),
        "matching": np.mean(matching_scores),
        "recall5": np.mean(recall5_scores)
    }