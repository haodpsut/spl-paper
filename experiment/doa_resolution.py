"""
Two-source resolution: where a learned equivariant model beats the classical
beamformer, while keeping the C_M rotational equivariance.

Single-snapshot UCA with two equal-power sources at azimuths phi_c +- delta/2 and a
random relative phase. The classical Bartlett beamformer cannot resolve two sources
much closer than its beamwidth; a learned model can exploit structure to resolve
smaller separations. We keep the cyclic-group C_M equivariance: rotating both
sources is a cyclic shift of the sensors, so the model uses circular convolutions
and a steerable angular-spectrum readout, now upsampled to a fine grid with a
cyclic (equivariant) upsampling so it can place sharp, super-resolving peaks.

We report the probability of resolution (both sources estimated within delta/2)
versus separation, on out-of-distribution azimuths (train phi_c in [0,pi),
test [pi,2pi)). Comments in English by project convention.
Run: python doa_resolution.py
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import doa_equivariance as D

M, ELEV_SIN, GAMMA = D.M, D.ELEV_SIN, D.GAMMA
# Larger radius than the single-source study: ~0.5 lambda inter-element spacing on
# the circle (circumference ~ M*0.5*lambda) gives kappa ~ 8, a realistic UCA with a
# usable beamwidth and no grating lobes, so two-source resolution lives in a sensible
# angular range. The C_M symmetry is unaffected by the value of kappa.
KR = 8.0
L = 128                      # fine angular grid (r = L/M = 8 bins per sensor step)
R = L // M
SNR_DB = 15.0
N_TRAIN = 4000
N_TEST = 2000
DELTAS_DEG = [8, 12, 16, 20, 28, 40]    # source separations to evaluate
EPOCHS = 60
BATCH = 256
LR = 2.5e-3
RESULTS_DIR = D.RESULTS_DIR
FINE = torch.tensor([2.0 * math.pi * b / L for b in range(L)], dtype=torch.float32)  # L-grid angles


# --------------------------------------------------------------------------
# Data: two sources, single snapshot
# --------------------------------------------------------------------------
def steer_np(phi):
    gamma = GAMMA.numpy()[None, :]
    return np.exp(1j * KR * ELEV_SIN * np.cos(phi[:, None] - gamma))


def circ_gauss(centers, sigma):
    """Target spectrum on the L-grid: sum of circular Gaussians at given angles."""
    ang = FINE.numpy()[None, :]                 # (1, L)
    t = np.zeros((centers.shape[0], L), dtype=np.float32)
    for k in range(centers.shape[1]):
        d = np.angle(np.exp(1j * (ang - centers[:, [k]])))
        t += np.exp(-0.5 * (d / sigma) ** 2)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def make_two_source(n, az_low, az_high, delta, snr_db, rng):
    phic = rng.uniform(az_low, az_high, size=n).astype(np.float32)
    phi1 = phic - delta / 2.0
    phi2 = phic + delta / 2.0
    a1, a2 = steer_np(phi1), steer_np(phi2)
    rel = np.exp(1j * rng.uniform(0, 2 * math.pi, size=(n, 1)))   # random relative phase
    sig = a1 + rel * a2
    snr_lin = 10.0 ** (snr_db / 10.0)
    psig = np.mean(np.abs(sig) ** 2)
    nstd = math.sqrt(psig / (2.0 * snr_lin))
    noise = nstd * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    obs = sig + noise
    obs = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-9)
    x = np.stack([obs.real, obs.imag], axis=1).astype(np.float32)
    centers = np.stack([phi1, phi2], axis=1).astype(np.float32)
    t = circ_gauss(centers, sigma=2.0 * math.pi / L * 1.5)        # ~1.5-bin wide peaks
    return (torch.from_numpy(x), torch.from_numpy(t),
            torch.from_numpy(phi1), torch.from_numpy(phi2))


# --------------------------------------------------------------------------
# Models: output an L-bin angular spectrum (sigmoid)
# --------------------------------------------------------------------------
# Fine-grid steering dictionary used as an equivariant front end.
def _steer_dict():
    grid = FINE.numpy()
    S = np.exp(1j * KR * ELEV_SIN * np.cos(grid[:, None] - GAMMA.numpy()[None, :]))  # (L, M)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    return S


class EquivRes(nn.Module):
    """C_M-equivariant super-resolution spectrum network (learned deconvolver).

    Front end: correlate the snapshot with the fine-grid steering dictionary, i.e.
    form the complex beamformer response b(phi_l) = a(phi_l)^H x at L angles. Under
    an azimuth rotation the snapshot shifts cyclically, so this response shifts by R
    bins, making the front end C_M-equivariant and giving the network fine angular
    resolution (no M-bottleneck). Circular convolutions then deconvolve the
    beampattern into sharp peaks; the output is L logits (peak-weighted BCE).
    """

    def __init__(self, ch=40, k=7):
        super().__init__()
        S = _steer_dict()
        self.register_buffer("Sr", torch.tensor(S.real, dtype=torch.float32))   # (L, M)
        self.register_buffer("Si", torch.tensor(S.imag, dtype=torch.float32))
        def cconv(ci, co): return nn.Conv1d(ci, co, k, padding=k // 2, padding_mode="circular")
        self.body = nn.Sequential(
            cconv(2, ch), nn.ReLU(), cconv(ch, ch), nn.ReLU(),
            cconv(ch, ch), nn.ReLU(), cconv(ch, ch), nn.ReLU(), cconv(ch, 1))

    def forward(self, x):
        xr, xi = x[:, 0], x[:, 1]                              # (B, M)
        # b = a^H x : real and imaginary parts of the fine-grid beamformer response.
        br = xr @ self.Sr.t() + xi @ self.Si.t()              # (B, L)
        bi = xi @ self.Sr.t() - xr @ self.Si.t()
        feat = torch.stack([br, bi], dim=1)                   # (B, 2, L)
        return self.body(feat).squeeze(1)                     # (B, L) logits


class MLPRes(nn.Module):
    """Non-equivariant counterpart: same steering-correlation front end, then fully
    connected layers (which break the cyclic equivariance). Outputs L logits."""

    def __init__(self, hidden=384):
        super().__init__()
        S = _steer_dict()
        self.register_buffer("Sr", torch.tensor(S.real, dtype=torch.float32))
        self.register_buffer("Si", torch.tensor(S.imag, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(2 * L, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, L))

    def forward(self, x):
        xr, xi = x[:, 0], x[:, 1]
        br = xr @ self.Sr.t() + xi @ self.Si.t()
        bi = xi @ self.Sr.t() - xr @ self.Si.t()
        feat = torch.cat([br, bi], dim=1)             # (B, 2L)
        return self.net(feat)                          # (B, L) logits


# --------------------------------------------------------------------------
# Peak picking and resolution metric
# --------------------------------------------------------------------------
def top2_peaks(spec):
    """Return the two largest circular local maxima (angles) of each spectrum row.

    Vectorized: mask non-maxima to -inf, then take the two highest bins per row.
    """
    s = spec
    left = np.roll(s, 1, axis=1); right = np.roll(s, -1, axis=1)
    ismax = (s >= left) & (s >= right)
    masked = np.where(ismax, s, -np.inf)
    idx2 = np.argpartition(-masked, 1, axis=1)[:, :2]   # two highest local maxima
    return FINE.numpy()[idx2].astype(np.float32)        # (n, 2) angles


def resolved(est2, phi1, phi2, tol):
    """Fraction of cases where both sources are matched within tol (greedy)."""
    def cdist(a, b): return np.abs(np.angle(np.exp(1j * (a - b))))
    e0, e1 = est2[:, 0], est2[:, 1]
    # two matchings, take the better
    m_a = np.maximum(cdist(e0, phi1), cdist(e1, phi2))
    m_b = np.maximum(cdist(e0, phi2), cdist(e1, phi1))
    worst = np.minimum(m_a, m_b)
    return float(np.mean(worst < tol))


def bartlett_spec(x):
    grid = FINE.numpy()
    S = np.exp(1j * KR * ELEV_SIN * np.cos(grid[:, None] - GAMMA.numpy()[None, :]))
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    obs = (x[:, 0] + 1j * x[:, 1]).numpy()
    return np.abs(obs @ S.conj().T) ** 2


def train_spec(model, xtr, ttr, augment, seed, epochs=EPOCHS):
    model.to(D.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Peak-weighted BCE: the target spectrum is sparse (two narrow peaks among L
    # bins), so a plain loss collapses to the all-zero output. pos_weight up-weights
    # the peak bins.
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(25.0))
    rng_t = torch.Generator().manual_seed(1000 + seed)
    n = xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=rng_t)
        model.train()
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, tb = xtr[idx], ttr[idx]
            if augment:
                # cyclic shift by sh sensors -> roll spectrum target by R*sh bins
                sh = int(torch.randint(0, M, (1,), generator=rng_t).item())
                xb = torch.roll(xb, shifts=sh, dims=2)
                tb = torch.roll(tb, shifts=R * sh, dims=1)
            xb, tb = xb.to(D.DEVICE), tb.to(D.DEVICE)
            loss = lossfn(model(xb), tb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def spec_resolution(model, x, phi1, phi2, tol):
    with torch.no_grad():
        s = torch.sigmoid(model(x.to(D.DEVICE))).cpu().numpy()
    return resolved(top2_peaks(s), phi1.numpy(), phi2.numpy(), tol)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------
# Full sweep
# --------------------------------------------------------------------------
SEEDS = [0, 1, 2]
DELTAS = [8, 12, 15, 18, 22, 30]
CONFIGS = [("Equivariant", "equiv", False), ("MLP+aug", "mlp", True)]


def mean_std(v):
    a = np.array(v, float)
    return float(a.mean()), float(a.std())


def main():
    res = {lab: {"ood": {}, "id": {}} for lab, _, _ in CONFIGS}
    res["Bartlett"] = {"ood": {}}
    params = {"Equivariant": count_params(EquivRes()), "MLP+aug": count_params(MLPRes())}

    for dd in DELTAS:
        delta = math.radians(dd); tol = delta / 2.0
        # Fixed test sets (independent of training seed).
        xid, _, p1i, p2i = make_two_source(N_TEST, 0.0, math.pi, delta, SNR_DB, np.random.default_rng(40 + dd))
        xood, _, p1o, p2o = make_two_source(N_TEST, math.pi, 2 * math.pi, delta, SNR_DB, np.random.default_rng(80 + dd))
        res["Bartlett"]["ood"][dd] = (resolved(top2_peaks(bartlett_spec(xood)), p1o.numpy(), p2o.numpy(), tol), 0.0)

        for lab, kind, aug in CONFIGS:
            ood_runs, id_runs = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed + dd)
                xtr, ttr, _, _ = make_two_source(N_TRAIN, 0.0, math.pi, delta, SNR_DB, rng)
                torch.manual_seed(seed)
                model = EquivRes() if kind == "equiv" else MLPRes()
                model = train_spec(model, xtr, ttr, aug, seed)
                ood_runs.append(spec_resolution(model, xood, p1o, p2o, tol))
                id_runs.append(spec_resolution(model, xid, p1i, p2i, tol))
            res[lab]["ood"][dd] = mean_std(ood_runs)
            res[lab]["id"][dd] = mean_std(id_runs)
            mo = res[lab]["ood"][dd][0]; mi = res[lab]["id"][dd][0]
            print(f"delta={dd:2d}  {lab:12s}  OOD={mo:.2f}  ID={mi:.2f}")
        print(f"delta={dd:2d}  Bartlett     OOD={res['Bartlett']['ood'][dd][0]:.2f}\n")

    out = {"config": {"M": M, "kr": KR, "L": L, "snr_db": SNR_DB, "n_train": N_TRAIN,
                      "deltas_deg": DELTAS, "seeds": SEEDS}, "params": params, "results": res}
    with open(os.path.join(RESULTS_DIR, "results_resolution.json"), "w") as f:
        json.dump({"config": out["config"], "params": params,
                   "results": {k: {kk: {str(d): vv for d, vv in dd.items()} for kk, dd in v.items()}
                               for k, v in res.items()}}, f, indent=2)
    print("Saved results_resolution.json  params:", params)
    make_figure(out); print_pgf(out)


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out["results"]; dg = out["config"]["deltas_deg"]
    plt.figure(figsize=(5.4, 3.7))
    plt.plot(dg, [res["Bartlett"]["ood"][d][0] for d in dg], "s--", color="#2E7D5B", label="Bartlett (classical)")
    for lab, color in [("MLP+aug", "#6B7280"), ("Equivariant", "#C8881C")]:
        ms = [res[lab]["ood"][d] for d in dg]
        plt.errorbar(dg, [m[0] for m in ms], yerr=[m[1] for m in ms], marker="o", capsize=2,
                     color=color, label=lab + " (OOD)", linewidth=1.7)
    eid = [res["Equivariant"]["id"][d][0] for d in dg]
    plt.plot(dg, eid, ":", color="#C8881C", alpha=0.7, label="Equivariant (ID)")
    plt.xlabel("Source separation (deg)"); plt.ylabel("Probability of resolution")
    plt.title("Two-source resolution (single snapshot)")
    plt.ylim(-0.03, 1.05); plt.legend(fontsize=7); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fig_resolution.png"), dpi=160); plt.close()
    print("Saved fig_resolution.png")


def print_pgf(out):
    res = out["results"]; dg = out["config"]["deltas_deg"]
    print("\n==== RESOLUTION (delta, prob mean, std) ====")
    print("Bartlett: " + " ".join(f"({d},{res['Bartlett']['ood'][d][0]:.2f})" for d in dg))
    for lab in ["MLP+aug", "Equivariant"]:
        print(f"{lab} OOD: " + " ".join(f"({d},{res[lab]['ood'][d][0]:.2f})+-(0,{res[lab]['ood'][d][1]:.2f})" for d in dg))
    print("Equivariant ID: " + " ".join(f"({d},{res['Equivariant']['id'][d][0]:.2f})" for d in dg))


if __name__ == "__main__":
    main()