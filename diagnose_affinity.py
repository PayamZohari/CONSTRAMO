"""
diagnose_affinity.py
====================
Diagnostic tool: prints per-omics affinity power and fused network quality.

Usage in main.py, after make_affinity:
    from diagnose_affinity import diagnose_affinity, diagnose_fused
    purity_scores = diagnose_affinity(train_affinity,
                                      labels_df['class'].iloc[x_train.index].values,
                                      n_patients=5, K=20)

And after snf.snf:
    diagnose_fused(fused_matrix, labels_df['class'].iloc[x_train.index].values, K=20)
"""

import numpy as np


def diagnose_affinity(affinity_list, labels_input, n_patients=5, K=20, seed=42):
    omic_names = ['cna', 'met', 'mrna', 'rppa']
    if hasattr(labels_input, 'numpy'):
        labels = labels_input.numpy()
    else:
        labels = np.array(labels_input)

    N = labels.shape[0]
    np.random.seed(seed)
    sample_idx = np.random.choice(N, size=min(n_patients, N), replace=False)

    print("\n" + "="*70)
    print("AFFINITY MATRIX DIAGNOSTICS")
    print("="*70)

    # -- 1. Global stats per omics -------------------------------------------
    print("\n[1] Global affinity stats per omics (training patients only)")
    print(f"{'Omics':>6}  {'Mean':>10}  {'Std':>10}  {'Max':>10}  {'Near-zero%':>11}  {'KNN purity':>10}")
    print("-"*70)

    purity_scores = {}
    for i, (name, A) in enumerate(zip(omic_names, affinity_list)):
        mask    = ~np.eye(N, dtype=bool)
        off     = A[mask]
        near_z  = (off < 1e-6).mean() * 100

        purities = []
        for p in range(N):
            row      = A[p].copy()
            row[p]   = -1
            knn      = np.argsort(row)[-K:]
            purities.append((labels[knn] == labels[p]).mean())
        purity = np.mean(purities)
        purity_scores[name] = purity

        print(f"{name:>6}  {off.mean():10.6f}  {off.std():10.6f}  {off.max():10.6f}  "
              f"{near_z:10.1f}%  {purity:10.4f}")

    # -- 2. Per-patient neighbor labels --------------------------------------
    print(f"\n[2] Top-{K} neighbors per omics for {len(sample_idx)} sampled patients")
    print("    label(affinity) — first 8 neighbors shown\n")
    for p in sample_idx:
        print(f"  Patient {p:3d}  label={labels[p]}")
        for name, A in zip(omic_names, affinity_list):
            row    = A[p].copy(); row[p] = -1
            knn    = np.argsort(row)[-K:][::-1]
            pairs  = [f"{labels[j]}({A[p,j]:.4f})" for j in knn[:8]]
            pur    = (labels[knn] == labels[p]).mean()
            print(f"    {name:>4}: purity={pur:.2f}  {' '.join(pairs)}")
        print()

    # -- 3. Ranking ----------------------------------------------------------
    print("[3] Omics ranking by KNN label purity")
    ranked = sorted(purity_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(ranked, 1):
        bar = '#' * int(score * 40)
        print(f"  {rank}. {name:>4}  {score:.4f}  {bar}")

    print("\n  purity > 0.8 -> strongly discriminative")
    print("  purity < 0.5 -> weak / noisy for this cancer type")

    # -- 4. Scale comparison -------------------------------------------------
    print("\n[4] Affinity scale comparison (max values per omics)")
    maxes = [affinity_list[i].max() for i in range(4)]
    max_of_max = max(maxes)
    for name, mx in zip(omic_names, maxes):
        ratio = mx / max_of_max if max_of_max > 0 else 0
        bar   = '#' * int(ratio * 30)
        print(f"  {name:>4}  max={mx:.6f}  relative scale={ratio:.3f}  {bar}")
    print("\n  NOTE: large scale differences mean one omics dominates SNF fusion")
    print("="*70 + "\n")

    return purity_scores


def diagnose_fused(fused_matrix, labels_input, K=20):
    """
    Call after snf.snf() to check if fusion improved label purity
    vs. the individual omics.

    Parameters
    ----------
    fused_matrix : np.ndarray (N, N)  — raw output of snf.snf(), before make_laplacian
    labels_input : array-like of length N
    """
    if hasattr(labels_input, 'numpy'):
        labels = labels_input.numpy()
    else:
        labels = np.array(labels_input)

    N = labels.shape[0]
    A = fused_matrix

    purities = []
    for p in range(N):
        row    = A[p].copy(); row[p] = -1
        knn    = np.argsort(row)[-K:]
        purities.append((labels[knn] == labels[p]).mean())
    purity = np.mean(purities)

    mask   = ~np.eye(N, dtype=bool)
    off    = A[mask]
    near_z = (off < 1e-6).mean() * 100

    print("\n[FUSED NETWORK] KNN label purity after SNF fusion:")
    print(f"  purity={purity:.4f}  mean={off.mean():.6f}  max={off.max():.6f}  near-zero={near_z:.1f}%")
    if purity > 0.62:
        print("  -> Fusion IMPROVED over best single omics (mrna=0.607)")
    else:
        print("  -> Fusion did NOT improve over best single omics — SNF scale issue likely")
    print()