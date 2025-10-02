import numpy as np
import tqdm


# %%
def semi_nmf_hals(
        X, rank, max_iter=200, tol=1e-4, subsample_svd=None,
        random_state=0, dtype=np.float32, verbose=False,
        # --- new knobs to reduce overlap in H ---
        l1_h=0.0,  # L1 penalty on H (>=0). e.g., 1e-3 * np.median(|X|)
        topk_per_feature=None,  # int or None. e.g., 1 or 2 keeps only top-k entries per column of H
        power_sharpen=None,  # float or None. e.g., 1.5 or 2.0
        ortho_lambda=0.0,  # small, e.g., 1e-3 to decorrelate rows of H
        renorm='l2'  # 'l2' or 'l1' row renorm of H each outer iter (rescales W accordingly)
):
    """
    Semi-NMF via HALS with options to reduce overlap in H:
      X ≈ W @ H, with W unconstrained and H >= 0.

    Overlap-reduction tools:
      - l1_h:      sparsifies H (elementwise soft-threshold in HALS step).
      - topk_per_feature: keep only the top-k entries per feature (column) -> hard competition.
      - power_sharpen:    raise H to a power >1 to sharpen supports (then renormalize).
      - ortho_lambda:     penalize off-diagonals of H H^T (light anti-correlation between rows).

    Returns
    -------
    W : (n_samples, rank)
    H : (rank, n_features)
    info : dict
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=dtype)
    n, m = X.shape
    r = int(rank)
    eps = np.finfo(dtype).eps

    # ---------- SVD-based initialization ----------
    if subsample_svd is not None and subsample_svd < n:
        idx = rng.choice(n, size=subsample_svd, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    U, S, Vt = U[:, :r], S[:r], Vt[:r, :]

    # H0 >= 0 from SVD loadings
    H = np.maximum(0, (S[:, None] * Vt))  # (r, m_subset or m)
    # If SVD used a row subset, Vt still spans feature space; we can proceed.

    # Compute W0 by ridge LS on full X
    HHT = H @ H.T
    ridge = 1e-6 * np.trace(HHT) / r
    HHT_reg = HHT + ridge * np.eye(r, dtype=dtype)
    W = (X @ H.T) @ np.linalg.inv(HHT_reg)

    # Initial residual
    R = X - W @ H
    prev_loss = np.linalg.norm(R, ord='fro') ** 2

    # Precompute helper for top-k mask building
    def _apply_topk(H, k):
        if k is None or k >= H.shape[0]:
            return H
        # keep top-k per column
        idx = np.argpartition(H, kth=H.shape[0] - k, axis=0)
        top_idx = idx[-k:, :]  # (k, m)
        mask = np.zeros_like(H, dtype=bool)
        # put_along_axis wants same shape
        np.put_along_axis(mask, top_idx, True, axis=0)
        H[~mask] = 0
        return H

    for it in tqdm.tqdm(range(1, max_iter + 1)):
        # Cyclic HALS updates
        for k in range(r):
            # add back kth component to residual
            R += np.outer(W[:, k], H[k, :])

            # Update w_k (unconstrained LS)
            hk = H[k, :]
            denom_w = float(hk @ hk) + eps
            wk = (R @ hk) / denom_w
            W[:, k] = wk

            # Update h_k (>=0) with optional L1
            wk = W[:, k]
            denom_h = float(wk @ wk) + eps
            numer = wk.T @ R  # shape (m,)

            if l1_h > 0.0:
                # Solve: min_h>=0 0.5||R - w h||^2 + l1_h * ||h||_1
                # Closed form per element: h = max(0, (numer - l1_h)/denom_h)
                hk_new = (numer - l1_h)
                # numeric stability
                hk_new = hk_new / denom_h
                H[k, :] = np.maximum(0, hk_new)
            else:
                H[k, :] = np.maximum(0, numer / denom_h)

            # subtract refreshed component
            R -= np.outer(W[:, k], H[k, :])

        # ----- optional post-steps on H to reduce overlap -----
        changed = False

        # (a) Top-k competition per feature
        if topk_per_feature is not None:
            _apply_topk(H, int(topk_per_feature))
            changed = True

        # (b) Power-sharpen supports (then renormalize rows later)
        if power_sharpen is not None and power_sharpen > 1.0:
            H = np.power(H, power_sharpen, dtype=dtype)
            changed = True

        # (c) Light orthogonality / anti-correlation push on rows of H
        if ortho_lambda > 0.0:
            # penalize off-diagonal of G = H H^T
            G = H @ H.T  # (r, r)
            G_off = G - np.diag(np.diag(G))
            # Gradient of 0.5 * ||offdiag(HH^T)||_F^2 wrt H is (offdiag(HH^T)) H
            H = H - (2.0 * ortho_lambda) * (G_off @ H)
            H = np.maximum(0, H)
            changed = True

        # Row renormalization of H (push scale into W to keep WH)
        if renorm in ('l2', 'l1'):
            if renorm == 'l2':
                s = np.linalg.norm(H, axis=1) + eps
            else:  # 'l1'
                s = np.sum(H, axis=1) + eps
            H = H / s[:, None]
            W = W * s[None, :]
            changed = True

        # If we modified H/W outside the in-loop residual algebra, rebuild residual
        if changed:
            R = X - W @ H

        # Convergence check
        loss = np.linalg.norm(R, ord='fro') ** 2
        rel_impr = (prev_loss - loss) / (prev_loss + eps)
        if verbose and (it % 10 == 0 or it == 1):
            print(f"iter {it:4d}  loss={loss:.4e}  rel_impr={rel_impr:.3e}")
        if rel_impr < tol and it > 10:
            break
        prev_loss = loss

    info = {"loss": float(loss), "n_iter": it}
    return W, H, info
