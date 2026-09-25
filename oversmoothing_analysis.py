from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def average_cosine_similarity(emb):

    emb = emb.detach().cpu().numpy()

    sim = cosine_similarity(emb)

    n = sim.shape[0]

    mask = ~np.eye(n, dtype=bool)

    return sim[mask].mean()



def average_pairwise_distance(emb):

    emb = emb.detach().cpu().numpy()

    return pdist(emb).mean()


def class_separation_score(emb, labels):

    emb = emb.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    sim = cosine_similarity(emb)

    within = []
    between = []

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):

            if labels[i] == labels[j]:
                within.append(sim[i,j])
            else:
                between.append(sim[i,j])

    return (
        np.mean(within),
        np.mean(between),
        np.mean(within)-np.mean(between)
    )

def embedding_rank(emb):

    emb = emb.detach().cpu().numpy()

    return np.linalg.matrix_rank(emb)