"""
Multi-snapshot two-source resolution: equivariant deconvolver vs MUSIC and Bartlett.

This is the strong baseline study for the IEEE SPL letter. With several snapshots a
subspace method (MUSIC) becomes the right classical competitor, so we compare against
it rather than against Bartlett alone. We highlight the regime where learning beats
the subspace method: coherent (fully correlated) sources, as arise from specular
multipath, where the signal covariance is rank deficient and MUSIC's noise subspace
is contaminated, so MUSIC fails to resolve. Bartlett never resolves below its
beamwidth in either regime.

The C_M-equivariant model deconvolves the conventional (Bartlett) angular spectrum
of the sample covariance. That spectrum is robust to source coherence (it is just a
quadratic form a(phi)^H R a(phi)) and shifts cyclically under array rotation, so the
front end stays equivariant; circular convolutions then sharpen it into resolved
peaks. The model is trained once over a range of separations and evaluated per
separation, on out-of-distribution azimuths.

Run: python doa_resolution_ms.py
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import doa_equivariance as D

M, ELEV_SIN, GAMMA = D.M, D.ELEV_SIN, D.GAMMA
KR = 8.0                       # UCA radius giving a usable beamwidth (no grating lobes)
L = 128                        # fine angular grid
R_UP = L // M
T_SNAP = 10                    # number of snapshots
SNR_DB = 10.0
N_TRAIN = 4000
N_TEST = 2000
DELTA_TRAIN = (8.0, 34.0)      # train over a range of separations (degrees)
DELTAS = [8, 12, 15, 18, 22, 30]
EPOCHS = 60
BATCH = 256
LR = 2.5e-3
SEEDS = [0, 1, 2]
RESULTS_DIR = D.RESULTS_DIR
FINE = torch.tensor([2.0 * math.pi * b / L for b in range(L)], dtype=torch.float32)
GRID = FINE.numpy()
A_GRID = np.exp(1j * KR * ELEV_SIN * np.cos(GRID[:, None] - GAMMA.numpy()[None, :]))  # (L, M)
A_GRID = A_GRID / np.linalg.norm(A_GRID, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Data: T snapshots, sample covariance, Bartlett-spectrum front end
# --------------------------------------------------------------------------
def steer(phi):
    return np.exp(1j * KR * ELEV_SIN * np.cos(phi[:, None] - GAMMA.numpy()[None, :]))


def circ_gauss(centers, sigma):
    t = np.zeros((centers.shape[0], L), dtype=np.float32)
    for k in range(centers.shape[1]):
        d = np.angle(np.exp(1j * (GRID[None, :] - centers[:, [k]])))
        t += np.exp(-0.5 * (d / sigma) ** 2)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def make_dataset(n, az_low, az_high, delta_spec, T, snr_db, coherent, rng):
    """Generate n two-source samples; return Bartlett spectra, covariances, targets."""
    phic = rng.uniform(az_low, az_high, size=n)
    if isinstance(delta_spec, tuple):
        delta = np.radians(rng.uniform(delta_spec[0], delta_spec[1], size=n))
    else:
        delta = np.full(n, math.radians(delta_spec))
    phi1 = (phic - delta / 2.0).astype(np.float32)
    phi2 = (phic + delta / 2.0).astype(np.float32)
    a1, a2 = steer(phi1), steer(phi2)                       # (n, M)

    snr_lin = 10.0 ** (snr_db / 10.0)
    bspec = np.empty((n, L), dtype=np.float32)
    Rs = np.empty((n, M, M), dtype=np.complex64)
    for i in range(n):
        s1 = (rng.standard_normal(T) + 1j * rng.standard_normal(T)) / math.sqrt(2)
        if coherent:
            phase = np.exp(1j * rng.uniform(0, 2 * math.pi))   # fixed relative phase
            s2 = s1 * phase                                    # fully coherent
        else:
            s2 = (rng.standard_normal(T) + 1j * rng.standard_normal(T)) / math.sqrt(2)
        X = np.outer(a1[i], s1) + np.outer(a2[i], s2)          # (M, T)
        npow = 1.0 / snr_lin
        X = X + math.sqrt(npow / 2) * (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T)))
        Rmat = (X @ X.conj().T) / T                            # (M, M)
        Rs[i] = Rmat
        p = np.real(np.einsum('lm,mk,lk->l', A_GRID.conj(), Rmat, A_GRID))  # a^H R a per angle
        bspec[i] = (p / (p.max() + 1e-9)).astype(np.float32)
    target = circ_gauss(np.stack([phi1, phi2], axis=1), sigma=2.0 * math.pi / L * 1.5)
    return (torch.from_numpy(bspec[:, None, :]), Rs,
            torch.from_numpy(phi1), torch.from_numpy(phi2), torch.from_numpy(target))


# --------------------------------------------------------------------------
# Equivariant deconvolver on the (cyclic) Bartlett spectrum
# --------------------------------------------------------------------------
class EquivResMS(nn.Module):
    def __init__(self, ch=40, k=7):
        super().__init__()
        def cc(ci, co): return nn.Conv1d(ci, co, k, padding=k // 2, padding_mode="circular")
        self.body = nn.Sequential(cc(1, ch), nn.ReLU(), cc(ch, ch), nn.ReLU(),
                                  cc(ch, ch), nn.ReLU(), cc(ch, ch), nn.ReLU(), cc(ch, 1))

    def forward(self, x):                 # x: (B, 1, L)
        return self.body(x).squeeze(1)    # (B, L) logits


# --------------------------------------------------------------------------
# Baselines and peak picking
# --------------------------------------------------------------------------
def music_spectra(Rs, k=2):
    """MUSIC pseudo-spectrum on the fine grid for a batch of covariances."""
    n = Rs.shape[0]
    out = np.empty((n, L), dtype=np.float32)
    for i in range(n):
        w, V = np.linalg.eigh(Rs[i])           # ascending eigenvalues
        En = V[:, : M - k]                      # noise subspace
        proj = A_GRID.conj() @ En               # (L, M-k)
        denom = np.sum(np.abs(proj) ** 2, axis=1) + 1e-9
        out[i] = (1.0 / denom).astype(np.float32)
    return out


def top2_peaks(spec):
    left = np.roll(spec, 1, axis=1); right = np.roll(spec, -1, axis=1)
    masked = np.where((spec >= left) & (spec >= right), spec, -np.inf)
    idx2 = np.argpartition(-masked, 1, axis=1)[:, :2]
    return GRID[idx2].astype(np.float32)


def resolved(est2, phi1, phi2, tol):
    def cd(a, b): return np.abs(np.angle(np.exp(1j * (a - b))))
    e0, e1 = est2[:, 0], est2[:, 1]
    worst = np.minimum(np.maximum(cd(e0, phi1), cd(e1, phi2)),
                       np.maximum(cd(e0, phi2), cd(e1, phi1)))
    return float(np.mean(worst < tol))


def train(model, x, t, seed, epochs=EPOCHS):
    model.to(D.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(25.0))
    g = torch.Generator().manual_seed(1000 + seed)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        model.train()
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, tb = x[idx].to(D.DEVICE), t[idx].to(D.DEVICE)
            loss = lossfn(model(xb), tb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def net_resolution(model, x, phi1, phi2, tol):
    with torch.no_grad():
        s = torch.sigmoid(model(x.to(D.DEVICE))).cpu().numpy()
    return resolved(top2_peaks(s), phi1.numpy(), phi2.numpy(), tol)


def mean_std(v):
    a = np.array(v, float); return float(a.mean()), float(a.std())


# --------------------------------------------------------------------------
# Main: both regimes, multi-seed
# --------------------------------------------------------------------------
def run_regime(coherent, tag):
    print(f"\n===== regime: {tag} =====")
    res = {"Bartlett": {}, "MUSIC": {}, "Equivariant": {"ood": {}, "id": {}}}

    # Train equivariant models (one per seed) on a range of separations.
    models = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        xtr, _, _, _, ttr = make_dataset(N_TRAIN, 0.0, math.pi, DELTA_TRAIN, T_SNAP, SNR_DB, coherent, rng)
        torch.manual_seed(seed)
        models.append(train(EquivResMS(), xtr, ttr, seed))
        print(f"  trained seed {seed}")

    for dd in DELTAS:
        delta = math.radians(dd); tol = delta / 2.0
        # Fixed OOD and ID test sets at this separation.
        xo, Ro, p1o, p2o, _ = make_dataset(N_TEST, math.pi, 2 * math.pi, dd, T_SNAP, SNR_DB,
                                            coherent, np.random.default_rng(700 + dd))
        xi, _, p1i, p2i, _ = make_dataset(N_TEST, 0.0, math.pi, dd, T_SNAP, SNR_DB,
                                          coherent, np.random.default_rng(300 + dd))
        res["Bartlett"][dd] = (resolved(top2_peaks(xo[:, 0].numpy()), p1o.numpy(), p2o.numpy(), tol), 0.0)
        res["MUSIC"][dd] = (resolved(top2_peaks(music_spectra(Ro)), p1o.numpy(), p2o.numpy(), tol), 0.0)
        ood = [net_resolution(m, xo, p1o, p2o, tol) for m in models]
        idd = [net_resolution(m, xi, p1i, p2i, tol) for m in models]
        res["Equivariant"]["ood"][dd] = mean_std(ood)
        res["Equivariant"]["id"][dd] = mean_std(idd)
        print(f"  d={dd:2d}  Bartlett={res['Bartlett'][dd][0]:.2f}  MUSIC={res['MUSIC'][dd][0]:.2f}"
              f"  Equiv(OOD)={res['Equivariant']['ood'][dd][0]:.2f}")
    return res


def main():
    out = {"config": {"M": M, "kr": KR, "L": L, "T": T_SNAP, "snr_db": SNR_DB,
                      "n_train": N_TRAIN, "deltas": DELTAS, "seeds": SEEDS,
                      "delta_train": DELTA_TRAIN},
           "uncorrelated": run_regime(False, "uncorrelated"),
           "coherent": run_regime(True, "coherent")}
    with open(os.path.join(RESULTS_DIR, "results_resolution_ms.json"), "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print("\nSaved results_resolution_ms.json")
    make_figure(out); print_pgf(out)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dg = out["config"]["deltas"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, reg, title in [(axes[0], "uncorrelated", "Uncorrelated sources"),
                           (axes[1], "coherent", "Coherent sources")]:
        r = out[reg]
        ax.plot(dg, [r["Bartlett"][d][0] for d in dg], "s--", color="#2E7D5B", label="Bartlett")
        ax.plot(dg, [r["MUSIC"][d][0] for d in dg], "D-.", color="#7B3FA0", label="MUSIC")
        m = [r["Equivariant"]["ood"][d] for d in dg]
        ax.errorbar(dg, [v[0] for v in m], yerr=[v[1] for v in m], marker="o", capsize=2,
                    color="#C8881C", label="Equivariant (ours)", linewidth=1.7)
        ax.set_title(f"{title} (T={out['config']['T']})"); ax.set_xlabel("Separation (deg)")
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Probability of resolution"); axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fig_resolution_ms.png"), dpi=160); plt.close()
    print("Saved fig_resolution_ms.png")


def print_pgf(out):
    dg = out["config"]["deltas"]
    for reg in ["uncorrelated", "coherent"]:
        r = out[reg]
        print(f"\n==== {reg} ====")
        print("Bartlett: " + " ".join(f"({d},{r['Bartlett'][d][0]:.2f})" for d in dg))
        print("MUSIC:    " + " ".join(f"({d},{r['MUSIC'][d][0]:.2f})" for d in dg))
        print("Equiv OOD:" + " ".join(f"({d},{r['Equivariant']['ood'][d][0]:.2f})+-(0,{r['Equivariant']['ood'][d][1]:.2f})" for d in dg))
        print("Equiv ID: " + " ".join(f"({d},{r['Equivariant']['id'][d][0]:.2f})" for d in dg))


if __name__ == "__main__":
    main()