"""Synthetic Potts/Gaussian/negative-binomial generators for HMRF benchmarks.

Author : Tao Zhu <zhutao@tjh.tjmu.edu.cn>
Created: 2026-10-02
Updated: 2026-10-20

Purpose
-------
Produces controlled ground-truth data for the simulation experiments:
Potts label fields (sampled with blocked Gibbs or an FFT-smoothed random
field), Gaussian emissions with orthonormal class means, and
negative-binomial count emissions with configurable intensity factors.

Inputs
------
Grid sizes, number of classes, smoothness/emission parameters and a seed.
No file I/O.

Outputs
-------
Label arrays ``(H, W)``; feature matrices ``(n, d)``; class means ``(K, d)``.

Conventions
-----------
* Randomness is confined to ``numpy.random.Generator`` objects created from
  the ``seed`` argument.
* ``gaussian_rf_domains`` uses the arg-max of smoothed Gaussian fields, which
  yields irregular domain boundaries suited to large grids.

Usage
-----
    from hmrf.simulate import gaussian_rf_domains, nb_emission_domains
"""

import numpy as np

def _neighbor_counts(labels, K):
    """Counts of each state in the 4-neighborhood (zero-padded at borders, no wrap-around). labels: (H,W) int"""
    H, W = labels.shape
    oh = np.zeros((H, W, K), dtype=np.int32)
    oh[np.arange(H)[:, None], np.arange(W)[None, :], labels] = 1
    cnt = np.zeros_like(oh)
    for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        s = np.roll(oh, sgn, axis=ax)
        if ax == 0:
            if sgn == 1: s[0, :, :] = 0
            else: s[-1, :, :] = 0
        else:
            if sgn == 1: s[:, 0, :] = 0
            else: s[:, -1, :] = 0
        cnt += s
    return cnt

def sample_potts(H, W, K, beta, sweeps=2000, rng=None, seed=0):
    """Checkerboard block Gibbs sampling of the Potts ground truth."""
    rng = rng or np.random.default_rng(seed)
    labels = rng.integers(0, K, size=(H, W))
    # checkerboard parity mask
    Y, X = np.mgrid[0:H, 0:W]
    parity = (Y + X) % 2
    for _ in range(sweeps):
        for p in (0, 1):
            mask = parity == p
            cnt = _neighbor_counts(labels, K)
            logp = beta * cnt
            g = -np.log(-np.log(rng.random((H, W, K)) + 1e-30))
            new = np.argmax(logp + g, axis=2)
            labels[mask] = new[mask]
    return labels

def gaussian_emission(labels, K, d, mu_scale=1.5, sigma=0.8, rng=None, seed=1):
    """y_i ~ N(mu_{x_i}, sigma^2 I), mu_k random orthonormal."""
    rng = rng or np.random.default_rng(seed)
    A = rng.standard_normal((d, K))
    mu = np.linalg.qr(A)[0].T * mu_scale  # K x d orthonormal row vectors
    n = labels.size
    y = mu[labels.ravel()] + sigma * rng.standard_normal((n, d))
    return y, mu

def gaussian_rf_domains(S, K, smooth_sigma=None, bias=0.3, rng=None, seed=0):
    """Generate spatial domains as the arg-max of FFT-smoothed Gaussian fields.

    Args:
        S: Grid side length (the domain is ``S x S``).
        K: Number of classes.
        smooth_sigma: Correlation length in pixels; defaults to ``0.12 * S``.
        bias: Amplitude of a per-class random offset that breaks ties.
        rng: Optional ``numpy.random.Generator``.
        seed: Seed used when ``rng`` is None.

    Returns:
        Integer label array of shape ``(S, S)`` with irregular boundaries.
    """
    rng = rng or np.random.default_rng(seed)
    if smooth_sigma is None:
        smooth_sigma = S * 0.12
    fields = rng.standard_normal((K, S, S))
    fy = np.fft.fftfreq(S)[:, None] * 2 * np.pi
    fx = np.fft.rfftfreq(S)[None, :] * 2 * np.pi
    ker = np.exp(-0.5 * (smooth_sigma ** 2) * (fx * fx + fy * fy))
    sm = np.fft.irfft2(np.fft.rfft2(fields, axes=(1, 2)) * ker[None], axes=(1, 2))
    sm = (sm - sm.mean(axis=(1, 2), keepdims=True)) / (sm.std(axis=(1, 2), keepdims=True) + 1e-8)
    if bias:
        sm = sm + rng.uniform(0.0, float(bias), size=(K, 1, 1))
    return np.argmax(sm, axis=0).astype(np.int64)

def nb_emission(labels, K, d, mean_lo=0.2, mean_hi=4.0, disp=1.0, rng=None, seed=1):
    """NB count emission: mu_{k,g} ~ lognormal; y ~ NB(mu, disp). Returns (y, mu)."""
    rng = rng or np.random.default_rng(seed)
    mu = rng.uniform(mean_lo, mean_hi, size=(K, d))
    n = labels.size
    lam = mu[labels.ravel()]
    p = disp / (disp + lam)
    y = rng.negative_binomial(disp, p)
    return y.astype(np.float32), mu

def nb_emission_domains(labels, d, factors, disp=1.0, base_scale=1.0, rng=None, seed=1):
    """Domain-specific intensity: mu_kg = factors[k] * base_g (log-uniform base)."""
    rng = rng or np.random.default_rng(seed)
    base = np.exp(rng.uniform(-0.6, 0.6, size=d)) * base_scale
    mu = np.outer(np.asarray(factors, dtype=np.float64), base)
    lam = mu[labels.ravel()]
    p = disp / (disp + lam)
    y = rng.negative_binomial(disp, p)
    return y.astype(np.float32), mu
