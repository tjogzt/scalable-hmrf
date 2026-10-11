"""Alpha-expansion graph-cut inference for the multi-class Potts HMRF (pymaxflow, Kolmogorov-Zabih construction).

Convention: a node on the source side (S, segment==1) => b=1 (switched to alpha); on the sink side (T, segment==0) => b=0 (keeps its original label).
Cut cost = sum of S->T edge capacities; a node on S pays the sink cap, on T pays the source cap.
After each step the global energy is recomputed from scratch, and accepted only if the energy decreases (guards against implementation bugs).
"""
import numpy as np
import maxflow

def potts_energy(labels, unary, beta, H, W):
    lab = np.asarray(labels)
    E = float(unary[np.arange(H)[:, None], np.arange(W)[None, :], lab].sum())
    diff = 0
    for ax in (0, 1):
        s = np.roll(lab, 1, axis=ax)  # upper/left neighbor, each undirected edge counted once
        if ax == 0:
            s[0, :] = -1
        else:
            s[:, 0] = -1
        diff += int(((s >= 0) & (s != lab)).sum())
    return E + beta * diff

def alpha_step(unary, lab, H, W, beta, alpha, max_aux=3):
    """One binary step of alpha-expansion. Returns new labels whose energy does not increase."""
    n = H * W
    g = maxflow.Graph[float]()
    g.add_nodes(n + 2 * n)  # reserve auxiliary nodes (at most 1 per undirected edge)
    flat_u = unary.reshape(n, -1)
    flat_lab = np.asarray(lab).ravel()
    d0 = flat_u[np.arange(n), flat_lab].astype(float)
    d1 = flat_u[np.arange(n), alpha].astype(float)
    for i in range(n):
        mn = min(d0[i], d1[i])
        g.add_tedge(i, d0[i] - mn, d1[i] - mn)
    aux_id = n
    for y in range(H):
        for x in range(W):
            i = y * W + x
            for dy, dx in ((0, 1), (1, 0)):
                ny, nx = y + dy, x + dx
                if ny >= H or nx >= W:
                    continue
                j = ny * W + nx
                a, b = int(lab[y, x]), int(lab[ny, nx])
                if a == b:
                    if alpha != a:
                        g.add_edge(i, j, beta, beta)
                else:
                    s1 = 1.0 if a != alpha else 0.0
                    s2 = 1.0 if alpha != b else 0.0
                    P = beta * (s1 + s2 - 1.0)
                    u_i = beta * (s2 - 1.0)
                    u_j = beta * (s1 - 1.0)
                    if u_i >= 0:
                        g.add_tedge(i, 0.0, u_i)
                    else:
                        g.add_tedge(i, -u_i, 0.0)
                    if u_j >= 0:
                        g.add_tedge(j, 0.0, u_j)
                    else:
                        g.add_tedge(j, -u_j, 0.0)
                    if P > 0:
                        e = aux_id; aux_id += 1
                        g.add_tedge(e, P, 0.0)
                        g.add_edge(e, i, P, 0.0)
                        g.add_edge(e, j, P, 0.0)
    g.maxflow()
    new = lab.copy()
    new = new.astype(np.int64)
    # pymaxflow: get_segment==0 means source side (b=1, switched to alpha), 1 means sink side (b=0, kept)
    new[(np.array([g.get_segment(i) for i in range(n)]) == 0).reshape(H, W)] = alpha
    if potts_energy(new, unary, beta, H, W) < potts_energy(lab, unary, beta, H, W):
        return new
    return lab.astype(np.int64)

def alpha_expansion(unary, beta, max_sweeps=10, seed=0):
    """unary: (H,W,K)."""
    H, W, K = unary.shape
    rng = np.random.default_rng(seed)
    lab = np.argmax(unary, axis=2).astype(np.int64)
    for sw in range(max_sweeps):
        E_before = potts_energy(lab, unary, beta, H, W)
        for alpha in rng.permutation(K):
            lab = alpha_step(unary, lab, H, W, beta, int(alpha))
        if potts_energy(lab, unary, beta, H, W) >= E_before - 1e-9:
            break
    return lab
