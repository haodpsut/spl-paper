"""
Rotation-equivariant DOA estimation on a uniform circular array (UCA).

This experiment tests, in a non-terrestrial-network (NTN) antenna context, how a
known geometric symmetry of the task is best exploited by a learned model. On a
UCA, rotating the source azimuth by the inter-sensor angle equals a cyclic shift
of the sensor index (group C_M), the simplest instance of the geometric-algebra /
Clifford equivariance principle.

We compare three ways of handling the symmetry:
  * ignore it          : MLP / Transformer trained on a restricted azimuth range;
  * augment for it     : the same models trained with random cyclic-shift
                         augmentation (the standard cheap alternative to
                         equivariance);
  * build it in        : a C_M-equivariant circular-convolution network.
A classical Bartlett beamformer is the non-learned reference.

Two studies, both over multiple seeds with mean and standard deviation:
  (A) data efficiency : out-of-distribution (OOD) azimuth error vs training size;
  (B) SNR robustness  : OOD error vs SNR at a fixed training size.

Models train on azimuths in [0, pi); we evaluate on held-out in-distribution (ID)
azimuths in [0, pi) and OOD azimuths in [pi, 2*pi). Comments are in English by
project convention. Run: python doa_equivariance.py
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------
M = 16                       # number of UCA sensors
KR = math.pi                 # (2*pi/lambda)*r with r = 0.5*lambda  ->  kr = pi
ELEV_SIN = 1.0               # sin(elevation); fixed so the signal depends on azimuth only
SEEDS = [0, 1, 2, 3, 4]      # independent training seeds
N_TEST = 2000
N_TRAIN_GRID = [200, 500, 1000, 2000, 5000]
SNR_GRID_DB = [0, 5, 10, 15, 20]
N_FOR_SNR = 1000             # fixed training size for the SNR sweep
TRAIN_SNR_DB = 10.0          # training SNR for the data-efficiency study
TEST_SNR_DB = 10.0
EPOCHS = 80
BATCH = 128
LR = 2e-3
DEVICE = torch.device("cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = torch.tensor([2.0 * math.pi * i / M for i in range(M)], dtype=torch.float32)


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def make_dataset(n, az_low, az_high, snr_db, rng):
    """Generate n single-snapshot UCA samples with azimuth in [az_low, az_high)."""
    phi = rng.uniform(az_low, az_high, size=n).astype(np.float32)
    gamma = GAMMA.numpy()[None, :]
    phase = KR * ELEV_SIN * np.cos(phi[:, None] - gamma)
    sig = np.exp(1j * phase)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_std = math.sqrt(1.0 / (2.0 * snr_lin))
    noise = noise_std * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    obs = sig + noise
    obs = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-9)
    x = np.stack([obs.real, obs.imag], axis=1).astype(np.float32)   # (n, 2, M)
    y = np.stack([np.cos(phi), np.sin(phi)], axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(phi)


def cyclic_augment(xb, yb, rng_t):
    """Random cyclic-shift augmentation.

    Rolling the sensor axis by s steps is exactly an azimuth rotation by
    2*pi*s/M (a_i(phi + 2*pi*s/M) = a_{i-s}(phi)), so we roll the input and rotate
    the target unit vector by the same angle. A single shift per batch suffices;
    over many epochs it covers the whole rotation group.
    """
    s = int(torch.randint(0, M, (1,), generator=rng_t).item())
    xa = torch.roll(xb, shifts=s, dims=2)
    delta = 2.0 * math.pi * s / M
    c, sn = math.cos(delta), math.sin(delta)
    ya = torch.stack([yb[:, 0] * c - yb[:, 1] * sn,
                      yb[:, 0] * sn + yb[:, 1] * c], dim=1)
    return xa, ya


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def unit(v, eps=1e-8):
    return v / (v.norm(dim=1, keepdim=True) + eps)


class MLP(nn.Module):
    """Plain MLP on the flattened signal. No rotation inductive bias."""

    def __init__(self, hidden=160):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * M, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return unit(self.net(x))


class TransformerDOA(nn.Module):
    """Self-attention over sensors with a fixed cyclic positional encoding."""

    def __init__(self, d_model=32, nhead=4, layers=2, ff=64):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        pos = torch.stack([torch.cos(GAMMA), torch.sin(GAMMA)], dim=1)
        self.register_buffer("pos", pos)
        self.pos_proj = nn.Linear(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                         batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        tok = x.transpose(1, 2)
        h = self.inp(tok) + self.pos_proj(self.pos)[None]
        h = self.enc(h).mean(dim=1)
        return unit(self.head(h))


class EquivCircConv(nn.Module):
    """C_M-equivariant circular-convolution network with a steerable readout."""

    def __init__(self, ch=32, k=5, blocks=3):
        super().__init__()
        layers = [nn.Conv1d(2, ch, k, padding=k // 2, padding_mode="circular"), nn.ReLU()]
        for _ in range(blocks - 1):
            layers += [nn.Conv1d(ch, ch, k, padding=k // 2, padding_mode="circular"), nn.ReLU()]
        layers += [nn.Conv1d(ch, 1, k, padding=k // 2, padding_mode="circular")]
        self.body = nn.Sequential(*layers)
        bins = torch.stack([torch.cos(GAMMA), torch.sin(GAMMA)], dim=1)
        self.register_buffer("bins", bins)

    def forward(self, x):
        logits = self.body(x).squeeze(1)
        w = torch.softmax(logits, dim=1)
        return unit(w @ self.bins)


def build(name):
    return {"MLP": MLP, "Transformer": TransformerDOA, "EquivCircConv": EquivCircConv}[name]()


# Configurations: (display label, model class name, use augmentation).
CONFIGS = [
    ("MLP", "MLP", False),
    ("MLP+aug", "MLP", True),
    ("Transformer", "Transformer", False),
    ("Transformer+aug", "Transformer", True),
    ("Equivariant", "EquivCircConv", False),
]


# --------------------------------------------------------------------------
# Classical reference and metric
# --------------------------------------------------------------------------
def bartlett_eval(x, phi_true, grid_size=720):
    grid = np.linspace(0.0, 2.0 * math.pi, grid_size, endpoint=False)
    gamma = GAMMA.numpy()[None, :]
    steer = np.exp(1j * KR * ELEV_SIN * np.cos(grid[:, None] - gamma))
    steer = steer / np.linalg.norm(steer, axis=1, keepdims=True)
    obs = (x[:, 0] + 1j * x[:, 1]).numpy()
    spectrum = np.abs(obs @ steer.conj().T) ** 2
    phi_hat = grid[np.argmax(spectrum, axis=1)]
    return ang_err_deg(phi_hat, phi_true.numpy())


def ang_err_deg(phi_hat, phi_true):
    d = np.angle(np.exp(1j * (phi_hat - phi_true)))
    return float(np.mean(np.abs(d)) * 180.0 / math.pi)


def pred_angle(model, x):
    with torch.no_grad():
        v = model(x.to(DEVICE)).cpu().numpy()
    return np.arctan2(v[:, 1], v[:, 0])


def train_model(model, xtr, ytr, augment, seed, epochs=EPOCHS):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng_t = torch.Generator().manual_seed(1000 + seed)   # augmentation randomness
    n = xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=rng_t)
        model.train()
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = xtr[idx], ytr[idx]
            if augment:
                xb, yb = cyclic_augment(xb, yb, rng_t)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = ((pred - yb) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def count_params(name):
    return sum(p.numel() for p in build(name).parameters())


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
def run_all():
    # Fixed evaluation sets (independent of training seed) for the main study.
    eval_rng = np.random.default_rng(9999)
    xid, _, pid = make_dataset(N_TEST, 0.0, math.pi, TEST_SNR_DB, eval_rng)
    xood, _, pood = make_dataset(N_TEST, math.pi, 2.0 * math.pi, TEST_SNR_DB, eval_rng)

    labels = [c[0] for c in CONFIGS]
    params = {c[0]: count_params(c[1]) for c in CONFIGS}

    # ---- Study A: data efficiency (OOD and ID error vs training size) ----
    eff = {lab: {"id": {}, "ood": {}} for lab in labels}
    bart_eff = {"id": bartlett_eval(xid, pid), "ood": bartlett_eval(xood, pood)}
    for n_train in N_TRAIN_GRID:
        for lab, cls, aug in CONFIGS:
            id_runs, ood_runs = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                xtr, ytr, _ = make_dataset(n_train, 0.0, math.pi, TRAIN_SNR_DB, rng)
                torch.manual_seed(seed)
                model = train_model(build(cls), xtr, ytr, aug, seed)
                id_runs.append(ang_err_deg(pred_angle(model, xid), pid.numpy()))
                ood_runs.append(ang_err_deg(pred_angle(model, xood), pood.numpy()))
            eff[lab]["id"][n_train] = mean_std(id_runs)
            eff[lab]["ood"][n_train] = mean_std(ood_runs)
            m, sd = eff[lab]["ood"][n_train]
            print(f"[A] N={n_train:5d}  {lab:16s}  OOD={m:6.2f}+/-{sd:4.2f} deg")

    # ---- Study B: SNR robustness (OOD error vs SNR at fixed training size) ----
    snr = {lab: {} for lab in labels}
    bart_snr = {}
    for snr_db in SNR_GRID_DB:
        xid_s, _, pid_s = make_dataset(N_TEST, 0.0, math.pi, snr_db, np.random.default_rng(7000 + snr_db))
        xood_s, _, pood_s = make_dataset(N_TEST, math.pi, 2.0 * math.pi, snr_db, np.random.default_rng(8000 + snr_db))
        bart_snr[snr_db] = bartlett_eval(xood_s, pood_s)
        for lab, cls, aug in CONFIGS:
            ood_runs = []
            for seed in SEEDS:
                rng = np.random.default_rng(100 + seed)
                xtr, ytr, _ = make_dataset(N_FOR_SNR, 0.0, math.pi, snr_db, rng)
                torch.manual_seed(seed)
                model = train_model(build(cls), xtr, ytr, aug, seed)
                ood_runs.append(ang_err_deg(pred_angle(model, xood_s), pood_s.numpy()))
            snr[lab][snr_db] = mean_std(ood_runs)
            m, sd = snr[lab][snr_db]
            print(f"[B] SNR={snr_db:3d}  {lab:16s}  OOD={m:6.2f}+/-{sd:4.2f} deg")

    out = {
        "config": {"M": M, "kr": KR, "seeds": SEEDS, "n_test": N_TEST,
                   "n_train_grid": N_TRAIN_GRID, "snr_grid_db": SNR_GRID_DB,
                   "n_for_snr": N_FOR_SNR, "train_snr_db": TRAIN_SNR_DB, "epochs": EPOCHS},
        "params": params,
        "bartlett_eff": bart_eff, "bartlett_snr": bart_snr,
        "data_efficiency": eff, "snr_sweep": snr,
    }
    with open(os.path.join(RESULTS_DIR, "results_full.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nParams:", params)
    print("Saved results_full.json")
    make_figures(out)
    print_pgf_tables(out)


# --------------------------------------------------------------------------
# Figures and pgfplots-ready dumps
# --------------------------------------------------------------------------
COLORS = {
    "MLP": "#6B7280", "MLP+aug": "#9CA3AF",
    "Transformer": "#1F5C8B", "Transformer+aug": "#7FA8C9",
    "Equivariant": "#C8881C",
}


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = out["config"]["n_train_grid"]
    eff = out["data_efficiency"]
    plt.figure(figsize=(5.4, 3.7))
    for lab in COLORS:
        ms = [eff[lab]["ood"][str(n)] if str(n) in eff[lab]["ood"] else eff[lab]["ood"][n] for n in grid]
        mean = [a[0] for a in ms]
        std = [a[1] for a in ms]
        plt.errorbar(grid, mean, yerr=std, marker="o", capsize=2, color=COLORS[lab], label=lab, linewidth=1.6)
    plt.axhline(out["bartlett_eff"]["ood"], color="#2E7D5B", linestyle="--", linewidth=1.3, label="Bartlett")
    plt.xscale("log"); plt.yscale("symlog", linthresh=2)
    plt.xlabel("Training set size"); plt.ylabel("OOD azimuth error (deg)")
    plt.title("Data efficiency: equivariance vs augmentation")
    plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fig_data_efficiency.png"), dpi=160); plt.close()

    snr = out["snr_sweep"]; sg = out["config"]["snr_grid_db"]
    plt.figure(figsize=(5.4, 3.7))
    for lab in COLORS:
        ms = [snr[lab][str(s)] if str(s) in snr[lab] else snr[lab][s] for s in sg]
        mean = [a[0] for a in ms]; std = [a[1] for a in ms]
        plt.errorbar(sg, mean, yerr=std, marker="o", capsize=2, color=COLORS[lab], label=lab, linewidth=1.6)
    plt.plot(sg, [out["bartlett_snr"][str(s)] if str(s) in out["bartlett_snr"] else out["bartlett_snr"][s] for s in sg],
             color="#2E7D5B", linestyle="--", linewidth=1.3, label="Bartlett")
    plt.yscale("symlog", linthresh=2)
    plt.xlabel("SNR (dB)"); plt.ylabel("OOD azimuth error (deg)")
    plt.title(f"SNR robustness (N={out['config']['n_for_snr']})")
    plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fig_snr.png"), dpi=160); plt.close()
    print("Saved figures.")


def print_pgf_tables(out):
    """Print pgfplots-ready coordinate strings (mean and std) for the paper."""
    grid = out["config"]["n_train_grid"]
    sg = out["config"]["snr_grid_db"]
    print("\n==== DATA EFFICIENCY (N, OOD mean, OOD std) ====")
    for lab in COLORS:
        coords = " ".join(f"({n},{out['data_efficiency'][lab]['ood'][n][0]:.2f})+-(0,{out['data_efficiency'][lab]['ood'][n][1]:.2f})"
                          for n in grid)
        print(f"{lab}: {coords}")
    print(f"Bartlett OOD eff: {out['bartlett_eff']['ood']:.2f}")
    print("\n==== SNR SWEEP (SNR, OOD mean, OOD std) ====")
    for lab in COLORS:
        coords = " ".join(f"({s},{out['snr_sweep'][lab][s][0]:.2f})+-(0,{out['snr_sweep'][lab][s][1]:.2f})" for s in sg)
        print(f"{lab}: {coords}")
    print("Bartlett OOD snr: " + " ".join(f"({s},{out['bartlett_snr'][s]:.2f})" for s in sg))
    print("\n==== ID (N=5000) and params ====")
    for lab in COLORS:
        m, sd = out["data_efficiency"][lab]["id"][grid[-1]]
        print(f"{lab}: ID={m:.2f}+/-{sd:.2f}  params={out['params'][lab]}")


if __name__ == "__main__":
    run_all()