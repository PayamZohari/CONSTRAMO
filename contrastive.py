"""
contrastive.py  -  Cross-Modal Relational Contrastive Loss for MO-GCAN
=======================================================================

The idea (from supervisor):
    Treat each omics as a modality. For patients i and j, the difference
    vector between their embeddings in modality m should be similar to the
    difference vector in modality n (same pair, different modality), and
    dissimilar to difference vectors of other patient pairs.

    d_m(i,j) = h_m_i - h_m_j

    Loss enforces: d_m(i,j) ~ d_n(i,j)  for all modality pairs (m,n)
                   d_m(i,j) != d_m(i,k)  for k != j

    This is an InfoNCE (NT-Xent) loss over relation vectors.
    No label information is used.
"""

import torch
import torch.nn.functional as F
import itertools


def _pairwise_diff_vectors(emb):
    """
    emb: (N, D)
    returns: (P, D) where P = N*(N-1)/2, upper triangle pairs i < j
    """
    N = emb.shape[0]
    rows, cols = [], []
    for i in range(N):
        for j in range(i + 1, N):
            rows.append(i)
            cols.append(j)
    rows = torch.tensor(rows, device=emb.device)
    cols = torch.tensor(cols, device=emb.device)
    return emb[rows] - emb[cols]


def cross_modal_relational_loss(embeddings, temperature=0.5, max_pairs=256):
    """
    Parameters
    ----------
    embeddings  : list of M tensors, each (N, D) - one per omics
    temperature : float - InfoNCE temperature
    max_pairs   : int   - cap on number of patient pairs (memory)

    Returns
    -------
    scalar loss tensor
    """
    M = len(embeddings)
    if M < 2:
        return torch.tensor(0.0, requires_grad=True)

    device = embeddings[0].device

    # build pairwise difference vectors for each modality
    all_diffs = [_pairwise_diff_vectors(emb) for emb in embeddings]
    P = all_diffs[0].shape[0]

    # subsample pairs if needed
    if P > max_pairs:
        perm = torch.randperm(P, device=device)[:max_pairs]
        all_diffs = [d[perm] for d in all_diffs]
        P = max_pairs

    # normalize to unit sphere
    all_diffs = [F.normalize(d, dim=-1) for d in all_diffs]

    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for m, n in itertools.combinations(range(M), 2):
        r_m = all_diffs[m]   # (P, D)
        r_n = all_diffs[n]   # (P, D)

        # similarity matrix (P, P)
        sim = torch.mm(r_m, r_n.T) / temperature

        # positive = diagonal (same patient pair, different modality)
        labels = torch.arange(P, device=device)

        # symmetric InfoNCE
        loss_mn = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) * 0.5
        total_loss = total_loss + loss_mn
        n_pairs += 1

    return total_loss / n_pairs
