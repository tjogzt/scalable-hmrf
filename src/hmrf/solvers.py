"""Mean-field variational inference and blocked Gibbs sampling for Potts HMRFs.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Purpose
-------
Implements the two inference engines used throughout the project:

* ``mean_field_potts`` / ``mean_field_graph`` -- fast coordinate-ascent
  mean-field updates for a Potts prior with Gaussian or generic unaries;
* ``checkerboard_gibbs`` -- exact blocked Gibbs sampler used as the reference
  (slow) baseline and for measuring VI speed-ups;
* ``em_mf_*`` -- variational EM variants that also estimate emission
  parameters (Gaussian mean/variance, negative-binomial mean with fixed
  dispersion), with optional multiple restarts;
* ``adjusted_rand_index`` -- clustering agreement used by every benchmark.

Inputs
------
In-memory NumPy arrays only (images ``(H, W, d)`` or feature matrices
``(n, d)``; unaries ``(H, W, K)`` or ``(n, K)``). No file I/O.

Outputs
-------
Posterior probability fields ``q``, hard labels, fitted emission parameters,
iteration counts and (for the Gaussian variants) the ELBO.

Conventions
-----------
* The grid variants use 4-neighbourhoods with zero padding (no wrap-around).
* The graph variants accept an arbitrary undirected edge list ``(m, 2)``.
* All random draws flow through ``numpy.random.Generator`` objects seeded by
  the caller; global RNG state is never touched.
* Log-space arithmetic subtracts the per-pixel maximum before exponentiation.

Usage
-----
    from hmrf.solvers import em_mf_nb_restarts, adjusted_rand_index
"""

import numpy as np

def _neighbor_sum_grid(q, K):
    """q: (H,W,K) probability field; returns the sum of q over each point's 4-neighborhood (same shape)."""
    s = np.zeros_like(q)
    for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        t = np.roll(q, sgn, axis=ax)
        if ax == 0:
            if sgn == 1: t[0, :, :] = 0.0
            else: t[-1, :, :] = 0.0
        else:
            if sgn == 1: t[:, 0, :] = 0.0
            else: t[:, -1, :] = 0.0
        s += t
    return s

def mean_field_potts(log_u, beta, max_iter=500, tol=1e-5):
    """Run mean-field coordinate ascent to convergence.

    Args:
        log_u: Log-unary field of shape ``(H, W, K)`` (higher is better).
        beta: Potts smoothness weight.
        max_iter: Maximum number of sweeps.
        tol: Convergence threshold on the largest probability change.

    Returns:
        Tuple ``(q, iterations)`` where ``q`` has shape ``(H, W, K)`` and rows
        sum to one.
    """
    H, W, K = log_u.shape
    log_u = log_u - log_u.max(axis=2, keepdims=True)
    q = np.exp(log_u)
    q /= q.sum(axis=2, keepdims=True)
    for it in range(max_iter):
        nbr = _neighbor_sum_grid(q, K)
        logq = log_u + beta * nbr
        logq -= logq.max(axis=2, keepdims=True)
        new = np.exp(logq)
        new /= new.sum(axis=2, keepdims=True)
        delta = np.abs(new - q).max()
        q = new
        if delta < tol:
            return q, it + 1
    return q, max_iter

def checkerboard_gibbs(log_u, beta, sweeps=2000, burn=500, rng=None, seed=0):
    """Block Gibbs sampling of the Potts posterior; returns the posterior mode labels and the per-sample labels (for mean ARI); timing is estimated by the caller."""
    rng = rng or np.random.default_rng(seed)
    H, W, K = log_u.shape
    Y, X = np.mgrid[0:H, 0:W]
    parity = (Y + X) % 2
    labels = np.argmax(log_u, axis=2)
    samples = []
    for s in range(sweeps):
        for p in (0, 1):
            mask = parity == p
            cnt = np.zeros((H, W, K))
            oh = np.zeros((H, W, K))
            oh[Y, X, labels] = 1.0
            for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                t = np.roll(oh, sgn, axis=ax)
                if ax == 0:
                    if sgn == 1: t[0, :, :] = 0.0
                    else: t[-1, :, :] = 0.0
                else:
                    if sgn == 1: t[:, 0, :] = 0.0
                    else: t[:, -1, :] = 0.0
                cnt += t
            logp = log_u + beta * cnt
            logp -= logp.max(axis=2, keepdims=True)
            pmat = np.exp(logp)
            pmat /= pmat.sum(axis=2, keepdims=True)
            g = -np.log(-np.log(rng.random((H, W, K)) + 1e-30))
            new = np.argmax(np.log(pmat + 1e-300) + g, axis=2)
            labels[mask] = new[mask]
        if s >= burn:
            samples.append(labels.copy())
    st = np.stack(samples)  # (S,H,W)
    counts = np.zeros((H, W, K), dtype=np.int64)
    for k in range(K):
        counts[:, :, k] = (st == k).sum(axis=0)
    mode = np.argmax(counts, axis=2)
    return mode, samples

def gaussian_log_unaries(y2d, mu, sigma):
    """y2d:(n,d) mu:(K,d) -> (n,K) Gaussian log-likelihood (constant term retained)."""
    d = y2d.shape[1]
    s2 = float(sigma) ** 2
    ysq = (y2d ** 2).sum(1, keepdims=True)
    return -0.5 * d * np.log(2 * np.pi * s2) - (ysq - 2 * y2d @ mu.T + (mu ** 2).sum(1)[None, :]) / (2 * s2)

def nb_log_unaries(y2d, mu, disp):
    """NB emission unary terms (drop the constant term independent of k; does not affect softmax/M-step)."""
    logm = np.log(np.asarray(mu, dtype=np.float64))
    logdm = np.log(disp + mu)
    return y2d @ (logm.T - logdm.T) + disp * (y2d.shape[1] * np.log(disp) - logdm.sum(1))[None, :]

def em_mf_gaussian(y2d, K, beta, H, W, max_iter=200, tol=1e-4, restarts=3, sigma_floor=0.2, seed=0, use_f32=False):
    """VI-EM jointly estimates mu/sigma and the label posterior (multiple restarts, best ELBO selected). Returns (q, mu, sigma, iters, elbo)."""
    rng = np.random.default_rng(seed)
    y2d32 = y2d.astype(np.float32) if use_f32 else y2d
    best = None
    for r in range(restarts):
        lab = rng.integers(0, K, size=(H, W))
        mu = np.zeros((K, y2d.shape[1]), dtype=np.float64)
        for _ in range(5):  # kmeans-lite hard EM
            for k in range(K):
                m = lab.ravel() == k
                mu[k] = y2d[m].mean(axis=0) if m.any() else rng.standard_normal(y2d.shape[1])
            log_u = gaussian_log_unaries(y2d, mu, 1.0).reshape(H, W, K)
            nbr = np.zeros((H, W, K))
            oh = np.zeros((H, W, K))
            Y, X = np.mgrid[0:H, 0:W]
            oh[Y, X, lab] = 1.0
            for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                t = np.roll(oh, sgn, axis=ax)
                if ax == 0:
                    if sgn == 1: t[0, :, :] = 0.0
                    else: t[-1, :, :] = 0.0
                else:
                    if sgn == 1: t[:, 0, :] = 0.0
                    else: t[:, -1, :] = 0.0
                nbr += t
            lab = np.argmax(log_u + beta * nbr, axis=2)
        log_u = gaussian_log_unaries(y2d, mu, 1.0).reshape(H, W, K)
        if use_f32:
            log_u = log_u.astype(np.float32)
        q = np.exp(log_u - log_u.max(axis=2, keepdims=True))
        q /= q.sum(axis=2, keepdims=True)
        sigma = 1.0
        for it in range(max_iter):
            nbr = _neighbor_sum_grid(q, K)
            logq = log_u + beta * nbr
            logq -= logq.max(axis=2, keepdims=True)
            new = np.exp(logq); new /= new.sum(axis=2, keepdims=True)
            dq = np.abs(new - q).max()
            q = new
            # M-step (memory-friendly: expand ||y-mu||^2 via the ysq identity)
            if use_f32:
                w = q.reshape(-1, K).astype(np.float32)
                mu = (w.T @ y2d32) / (w.sum(0)[:, None] + 1e-12)
                ysq = (y2d32 ** 2).sum(1)
                sig2 = (w.T @ ysq) / (w.sum(0) * y2d32.shape[1] + 1e-12) - (mu ** 2).sum(1) / y2d32.shape[1]
                sigma = float(np.sqrt(max(sig2.max(), sigma_floor ** 2)))
                log_u = gaussian_log_unaries(y2d32, mu.astype(np.float64), sigma).reshape(H, W, K).astype(np.float32)
            else:
                w = q.reshape(-1, K).astype(np.float64)
                mu = (w.T @ y2d) / (w.sum(0)[:, None] + 1e-12)
                ysq = (y2d ** 2).sum(1)
                sig2 = (w.T @ ysq) / (w.sum(0) * y2d.shape[1] + 1e-12) - (mu ** 2).sum(1) / y2d.shape[1]
                sigma = float(np.sqrt(max(sig2.max(), sigma_floor ** 2)))
                log_u = gaussian_log_unaries(y2d, mu, sigma).reshape(H, W, K)
            if dq < tol and it > 10:
                break
        # ELBO = E_q[log p(y|z)] + E_q[log p(z)] + H[q]; used to select the restart
        e_logp = (q.reshape(-1, K) * log_u.reshape(-1, K)).sum()
        nbr = _neighbor_sum_grid(q, K)
        e_prior = 0.5 * beta * (q * nbr).sum()
        ent = -(q * np.log(q + 1e-300)).sum()
        elbo = e_logp + e_prior + ent
        if best is None or elbo > best[0]:
            best = (elbo, q, mu, sigma, it + 1)
    return best[1], best[2], best[3], best[4], best[0]

def em_mf_nb(y2d, K, beta, H, W, disp=1.0, max_iter=150, tol=1e-4, seed=0, rng=None, init_log1p=True):
    """NB-emission VI-EM (disp fixed); returns (q, mu, iters)."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    lab = rng.integers(0, K, size=(H, W))
    mu = np.zeros((K, y2d.shape[1]), dtype=np.float64)
    yt = y2d
    if init_log1p:
        yt = np.log1p(y2d)
        yt = (yt - yt.mean(0, keepdims=True)) / (yt.std(0, keepdims=True) + 1e-6)
    for _ in range(5):
        for k in range(K):
            m = lab.ravel() == k
            mu[k] = yt[m].mean(axis=0) if m.any() else rng.standard_normal(yt.shape[1])
        log_u = nb_log_unaries(y2d, mu, disp).reshape(H, W, K)
        nbr = np.zeros((H, W, K))
        oh = np.zeros((H, W, K)); Y, X = np.mgrid[0:H, 0:W]; oh[Y, X, lab] = 1.0
        for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            t = np.roll(oh, sgn, axis=ax)
            if ax == 0:
                if sgn == 1: t[0, :, :] = 0.0
                else: t[-1, :, :] = 0.0
            else:
                if sgn == 1: t[:, 0, :] = 0.0
                else: t[:, -1, :] = 0.0
            nbr += t
        lab = np.argmax(log_u + beta * nbr, axis=2)
    # After hard EM, re-estimate mu from the raw counts (initialization only shapes the labels)
    for k in range(K):
        m = lab.ravel() == k
        mu[k] = y2d[m].mean(axis=0) if m.any() else 0.5 + rng.random(y2d.shape[1])
    log_u = nb_log_unaries(y2d, mu, disp).reshape(H, W, K)
    q = np.exp(log_u - log_u.max(axis=2, keepdims=True)); q /= q.sum(axis=2, keepdims=True)
    for it in range(max_iter):
        nbr = _neighbor_sum_grid(q, K)
        logq = log_u + beta * nbr
        logq -= logq.max(axis=2, keepdims=True)
        new = np.exp(logq); new /= new.sum(axis=2, keepdims=True)
        dq = np.abs(new - q).max()
        q = new
        w = q.reshape(-1, K).astype(np.float64)
        mu = (w.T @ y2d) / (w.sum(0)[:, None] + 1e-12)
        log_u = nb_log_unaries(y2d, mu, disp).reshape(H, W, K)
        if dq < tol and it > 10:
            break
    nbr = _neighbor_sum_grid(q, K)
    obj = float((q * log_u).sum() + 0.5 * beta * (q * nbr).sum() - (q * np.log(q + 1e-300)).sum())
    return q, mu, it + 1, obj

def em_mf_nb_restarts(y2d, K, beta, H, W, disp=1.0, max_iter=300, tol=1e-3, restarts=3, seed=0):
    """Multi-restart NB VI-EM, best selected by the objective value. Returns (q, mu, iters, obj)."""
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        q, mu, iters, obj = em_mf_nb(y2d, K, beta, H, W, disp=disp, max_iter=max_iter,
                                     tol=tol, seed=seed + r, rng=rng, init_log1p=True)
        if best is None or obj > best[0]:
            best = (obj, q, mu, iters)
    return best[1], best[2], best[3], best[0]

# ---------- graph-structured version (arbitrary adjacency, for real-data grids with holes) ----------

def _nbr_sum_edges(q2d, edges):
    """q2d:(n,K); returns the sum of neighbor q per node. edges: (E,2) directed (both directions already included)."""
    n, K = q2d.shape
    out = np.zeros_like(q2d)
    np.add.at(out, edges[:, 0], q2d[edges[:, 1]])
    return out

def mean_field_graph(log_u2d, beta, edges, max_iter=500, tol=1e-5):
    log_u2d = log_u2d - log_u2d.max(axis=1, keepdims=True)
    q = np.exp(log_u2d); q /= q.sum(axis=1, keepdims=True)
    for it in range(max_iter):
        nbr = _nbr_sum_edges(q, edges)
        logq = log_u2d + beta * nbr
        logq -= logq.max(axis=1, keepdims=True)
        new = np.exp(logq); new /= new.sum(axis=1, keepdims=True)
        delta = np.abs(new - q).max()
        q = new
        if delta < tol:
            break
    return q, it + 1

def em_mf_gaussian_graph(y2d, K, beta, edges, max_iter=300, tol=1e-4, restarts=3,
                         sigma_floor=0.2, seed=0, init_labels=None):
    """Graph-structured VI-EM (Gaussian emission); returns (q, mu, sigma, iters, elbo). init_labels can be passed to fix the starting point."""
    n, d = y2d.shape
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        lab = np.asarray(init_labels, dtype=np.int64).copy() if init_labels is not None else rng.integers(0, K, size=n)
        mu = np.zeros((K, d), dtype=np.float64)
        if init_labels is None:  # hard-EM shaping only for random starts; external starts go straight to soft EM
            for _ in range(5):
                for k in range(K):
                    m = lab == k
                    mu[k] = y2d[m].mean(axis=0) if m.any() else rng.standard_normal(d)
                log_u = gaussian_log_unaries(y2d, mu, 1.0)
                nbr = np.zeros((n, K))
                oh = np.zeros((n, K)); oh[np.arange(n), lab] = 1.0
                nbr = _nbr_sum_edges(oh, edges)
                lab = np.argmax(log_u + beta * nbr, axis=1)
        for k in range(K):
            m = lab == k
            mu[k] = y2d[m].mean(axis=0) if m.any() else rng.standard_normal(d)
        log_u = gaussian_log_unaries(y2d, mu, 1.0)
        sigma_cap = float(np.sqrt((y2d ** 2).mean())) if (y2d ** 2).mean() > 0 else 1.0  # within-class variance <= total variance
        q = np.exp(log_u - log_u.max(axis=1, keepdims=True)); q /= q.sum(axis=1, keepdims=True)
        sigma = 1.0
        for it in range(max_iter):
            nbr = _nbr_sum_edges(q, edges)
            logq = log_u + beta * nbr
            logq -= logq.max(axis=1, keepdims=True)
            new = np.exp(logq); new /= new.sum(axis=1, keepdims=True)
            dq = np.abs(new - q).max()
            q = new
            w = q.astype(np.float64)
            mu = (w.T @ y2d) / (w.sum(0)[:, None] + 1e-12)
            ysq = (y2d ** 2).sum(1)
            sig2 = (w.T @ ysq) / (w.sum(0) * d + 1e-12) - (mu ** 2).sum(1) / d
            sigma = float(np.sqrt(max(sig2.max(), sigma_floor ** 2)))
            sigma = min(sigma, sigma_cap)
            log_u = gaussian_log_unaries(y2d, mu, sigma)
            if dq < tol and it > 10:
                break
        elbo = float((q * log_u).sum() + 0.5 * beta * (q * nbr).sum() - (q * np.log(q + 1e-300)).sum())
        if best is None or elbo > best[0]:
            best = (elbo, q, mu, sigma, it + 1)
    return best[1], best[2], best[3], best[4], best[0]

def adjusted_rand_index(labels_true, labels_pred):
    """Standard ARI (contingency-based implementation)."""
    from collections import Counter
    a = np.asarray(labels_true).ravel(); b = np.asarray(labels_pred).ravel()
    ct = Counter(zip(a.tolist(), b.tolist()))
    ca = Counter(a.tolist()); cb = Counter(b.tolist())
    n = a.size
    idx = sum(v * (v - 1) / 2 for v in ct.values())
    sa = sum(v * (v - 1) / 2 for v in ca.values())
    sb = sum(v * (v - 1) / 2 for v in cb.values())
    exp = sa * sb / (n * (n - 1) / 2)
    mx = (sa + sb) / 2
    if mx == exp:
        return 1.0 if idx == exp else 0.0
    return (idx - exp) / (mx - exp)
