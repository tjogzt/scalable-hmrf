"""高保真空间域仿真：Potts 真值 + 高斯/NB 发射（综述建议最先做的一件事①）。"""
import numpy as np

def _neighbor_counts(labels, K):
    """4-邻域各状态计数（边界补 0，不环绕）。labels: (H,W) int"""
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
    """Checkerboard 块 Gibbs 采样 Potts 真值。"""
    rng = rng or np.random.default_rng(seed)
    labels = rng.integers(0, K, size=(H, W))
    # 棋盘奇偶掩码
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
    """y_i ~ N(mu_{x_i}, sigma^2 I)，mu_k 随机标准正交。"""
    rng = rng or np.random.default_rng(seed)
    A = rng.standard_normal((d, K))
    mu = np.linalg.qr(A)[0].T * mu_scale  # K x d 正交行向量
    n = labels.size
    y = mu[labels.ravel()] + sigma * rng.standard_normal((n, d))
    return y, mu

def gaussian_rf_domains(S, K, smooth_sigma=None, bias=0.3, rng=None, seed=0):
    """FFT 平滑高斯随机场 argmax 生成空间域。smooth_sigma 默认 S*0.12（特征~12% 网格）。"""
    """FFT 平滑高斯随机场 argmax 生成空间域（不规则边界，适合大网格）。"""
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
    """NB 计数发射：mu_{k,g} ~ lognormal；y ~ NB(mu, disp)。返回 (y, mu)。"""
    rng = rng or np.random.default_rng(seed)
    mu = rng.uniform(mean_lo, mean_hi, size=(K, d))
    n = labels.size
    lam = mu[labels.ravel()]
    p = disp / (disp + lam)
    y = rng.negative_binomial(disp, p)
    return y.astype(np.float32), mu

def nb_emission_domains(labels, d, factors, disp=1.0, base_scale=1.0, rng=None, seed=1):
    """域特异强度：mu_kg = factors[k] * base_g（log-uniform base）。"""
    rng = rng or np.random.default_rng(seed)
    base = np.exp(rng.uniform(-0.6, 0.6, size=d)) * base_scale
    mu = np.outer(np.asarray(factors, dtype=np.float64), base)
    lam = mu[labels.ravel()]
    p = disp / (disp + lam)
    y = rng.negative_binomial(disp, p)
    return y.astype(np.float32), mu
