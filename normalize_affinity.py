"""
normalize_affinity.py
=====================
Scale-normalize affinity matrices before SNF fusion.

MOTIVATION (from diagnostic output)
-------------------------------------
For SARC, rppa affinity max = 0.009 while mrna max = 0.00006 — a 163x
difference. In the SNF update equation (eq. 11 in MO-GCAN paper):

    P^(m) = S^m * ( mean_{k != m} P^(k) ) * (S^m)^T

the averaged P matrices from other omics are multiplied against S^m.
When one omics has values 163x larger, it numerically dominates the
iterative updates regardless of its label discriminability.

mrna has the best KNN label purity (0.607) but near-zero absolute values,
so it contributes almost nothing to the fused network.
rppa has lower purity (0.542) but large absolute values, so it dominates.

FIX: normalize each affinity matrix A_m so that max(A_m) = 1 before SNF.
This makes the fusion a true average of neighborhood structures, not an
average dominated by numerical scale.

This is applied AFTER make_affinity and BEFORE snf.snf.
It does not change the topology (who is connected to whom), only the scale.
"""

import numpy as np


def normalize_affinity_scale(affinity_list, method='max'):
    """
    Normalize a list of affinity matrices to the same scale.

    Parameters
    ----------
    affinity_list : list of np.ndarray, shape (N, N)
        Output of snf.compute.make_affinity(...)
    method : str
        'max'  — divide each matrix by its own maximum value.
                 All matrices end up with max = 1.
                 Preserves relative within-omics structure exactly.
        'mean' — divide each matrix by its mean non-zero value.
                 Useful if one omics has a few extreme outlier edges.

    Returns
    -------
    normalized : list of np.ndarray, same shapes
    scale_factors : list of float (what each matrix was divided by)
    """
    omic_names = ['cna', 'met', 'mrna', 'rppa']
    normalized     = []
    scale_factors  = []

    print("\n[Affinity normalization]")
    for i, (name, A) in enumerate(zip(omic_names, affinity_list)):
        if method == 'max':
            factor = A.max()
        elif method == 'mean':
            nonzero = A[A > 1e-12]
            factor  = nonzero.mean() if len(nonzero) > 0 else 1.0
        else:
            raise ValueError(f"Unknown method: {method}")

        if factor < 1e-12:
            factor = 1.0   # avoid division by zero for degenerate omics

        A_norm = A / factor
        normalized.append(A_norm)
        scale_factors.append(factor)
        print(f"  {name:>4}: divided by {factor:.6f}  "
              f"(new max={A_norm.max():.4f}, new mean={A_norm.mean():.6f})")

    print()
    return normalized, scale_factors