"""
Hard regime for the IEEE letter: DOA under unknown circulant mutual coupling.

Motivation. On the clean single-source UCA, a classical Bartlett beamformer already
matches every learned model, so learning adds nothing. We therefore introduce a
realistic model mismatch, mutual coupling, under which the nominal classical
beamformer (which assumes the ideal steering vector) degrades, and ask whether a
learned model that does not know the coupling can recover the accuracy while still
generalizing across array orientations.

Key design point. For a UCA the mutual-coupling matrix is circulant (the coupling
between two sensors depends only on their angular separation). A circulant C
commutes with the cyclic shift, so the observation

    x = C a(phi) + noise

still shifts cyclically when the azimuth rotates, i.e. the cyclic symmetry C_M is
preserved. The C_M-equivariant model is thus correctly specified even under
coupling, and can learn to undo C implicitly. We compare:

  * Bartlett-nominal : classical beamformer with the ideal steering a(phi)  (unaware of C)
  * Bartlett-oracle  : classical beamformer with the true steering C a(phi) (knows C, upper bound)
  * Equivariant      : C_M-equivariant circular-conv net (does not know C)
  * MLP+aug, Transformer+aug : augmented non-equivariant baselines

We sweep the coupling strength and report OOD azimuth error (train [0,pi),
test [pi,2pi)) over 5 seeds. Comments in English by project convention.
Run: python doa_coupling.py
"""

import json
import math
import os

import numpy as np
import torch

import doa_equivariance as D   # reuse models, training, metric, constants

M, KR, ELEV_SIN, GAMMA = D.M, D.KR, D.ELEV_SIN, D.GAMMA
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 2000
N_TEST = 2000
SNR_DB = 10.0
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]   # nearest-neighbour coupling magnitude
EPOCHS = 80
RESULTS_DIR = D.RESULTS_DIR

# Learned configurations to evaluate (label, model class, augment).
LEARNED = [
    ("MLP+aug", "MLP", True),
    ("Transformer+aug", "Transformer", True),
    ("Equivariant", "EquivCircConv", False),
]


# --------------------------------------------------------------------------
# Circulant mutual-coupling model
# --------------------------------------------------------------------------
def coupling_matrix(alpha):
    """Symmetric circulant coupling matrix with decaying neighbour terms.

    c[0]=1; first neighbour magnitude alpha, second 0.4*alpha, with fixed phases.
    Symmetric (c[d]=c[M-d]) so C is Hermitian-circulant, a standard UCA model.
    """
    c = np.zeros(M, dtype=complex)
    c[0] = 1.0
    if M >= 3 and alpha > 0:
        # Reciprocal coupling: C is symmetric (c[d]=c[M-d]) but complex, hence
        # non-Hermitian, so its eigenvalues are complex and the nominal beamformer
        # peak is biased. A Hermitian (conjugate-symmetric) C would give real
        # eigenvalues, a pure taper that leaves a single-source peak in place.
        c[1] = alpha * np.exp(1j * math.pi / 3); c[M - 1] = c[1]
        c[2] = 0.45 * alpha * np.exp(1j * 0.9 * math.pi); c[M - 2] = c[2]
    C = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            C[i, j] = c[(i - j) % M]
    return C


def steering(phi):
    """Ideal UCA steering matrix a(phi): shape (len(phi), M)."""
    gamma = GAMMA.numpy()[None, :]
    return np.exp(1j * KR * ELEV_SIN * np.cos(phi[:, None] - gamma))


def make_coupled_dataset(n, az_low, az_high, snr_db, C, rng):
    """Single-snapshot UCA data with mutual coupling x = C a(phi) + noise."""
    phi = rng.uniform(az_low, az_high, size=n).astype(np.float32)
    a = steering(phi)                          # (n, M) ideal response
    sig = a @ C.T                              # apply coupling (circulant)
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Normalise noise to the average coupled-signal power so SNR is well defined.
    psig = np.mean(np.abs(sig) ** 2)
    noise_std = math.sqrt(psig / (2.0 * snr_lin))
    noise = noise_std * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    obs = sig + noise
    obs = obs / (np.linalg.norm(obs, axis=1, keepdims=True) + 1e-9)
    x = np.stack([obs.real, obs.imag], axis=1).astype(np.float32)
    y = np.stack([np.cos(phi), np.sin(phi)], axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(phi)


def bartlett_steered(x, phi_true, steer_grid):
    """Bartlett DOA with a supplied (assumed) steering dictionary steer_grid (G,M)."""
    steer = steer_grid / np.linalg.norm(steer_grid, axis=1, keepdims=True)
    obs = (x[:, 0] + 1j * x[:, 1]).numpy()
    spectrum = np.abs(obs @ steer.conj().T) ** 2
    grid = np.linspace(0.0, 2.0 * math.pi, steer_grid.shape[0], endpoint=False)
    return D.ang_err_deg(grid[np.argmax(spectrum, axis=1)], phi_true.numpy())


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------
def main():
    grid = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    a_grid = steering(grid)                       # ideal steering on the fine grid

    results = {lab: {} for lab, _, _ in LEARNED}
    results["Bartlett-nominal"] = {}
    results["Bartlett-oracle"] = {}

    for alpha in ALPHA_GRID:
        C = coupling_matrix(alpha)
        nominal_grid = a_grid                     # assumes no coupling
        oracle_grid = a_grid @ C.T                # knows the true coupling

        # Fixed OOD test set for this alpha (independent of training seed).
        xood, _, pood = make_coupled_dataset(N_TEST, math.pi, 2 * math.pi, SNR_DB,
                                              C, np.random.default_rng(5000 + int(alpha * 100)))
        results["Bartlett-nominal"][alpha] = (bartlett_steered(xood, pood, nominal_grid), 0.0)
        results["Bartlett-oracle"][alpha] = (bartlett_steered(xood, pood, oracle_grid), 0.0)

        for lab, cls, aug in LEARNED:
            runs = []
            for seed in SEEDS:
                rng = np.random.default_rng(200 + seed + int(alpha * 100))
                xtr, ytr, _ = make_coupled_dataset(N_TRAIN, 0.0, math.pi, SNR_DB, C, rng)
                torch.manual_seed(seed)
                model = D.train_model(D.build(cls), xtr, ytr, aug, seed, epochs=EPOCHS)
                runs.append(D.ang_err_deg(D.pred_angle(model, xood), pood.numpy()))
            results[lab][alpha] = D.mean_std(runs)
            m, sd = results[lab][alpha]
            print(f"alpha={alpha:.2f}  {lab:16s}  OOD={m:6.2f}+/-{sd:4.2f} deg")
        bn = results["Bartlett-nominal"][alpha][0]
        bo = results["Bartlett-oracle"][alpha][0]
        print(f"alpha={alpha:.2f}  Bartlett-nominal OOD={bn:6.2f}   Bartlett-oracle OOD={bo:6.2f}\n")

    out = {"config": {"M": M, "n_train": N_TRAIN, "snr_db": SNR_DB,
                      "alpha_grid": ALPHA_GRID, "seeds": SEEDS, "epochs": EPOCHS},
           "results": results}
    with open(os.path.join(RESULTS_DIR, "results_coupling.json"), "w") as f:
        json.dump({k: {str(a): v for a, v in d.items()} for k, d in results.items()}
                  | {"_config": out["config"]}, f, indent=2)
    print("Saved results_coupling.json")
    make_figure(out)
    print_pgf(out)


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = out["results"]; ag = out["config"]["alpha_grid"]
    colors = {"MLP+aug": "#6B7280", "Transformer+aug": "#1F5C8B", "Equivariant": "#C8881C",
              "Bartlett-nominal": "#B23A3A", "Bartlett-oracle": "#2E7D5B"}
    plt.figure(figsize=(5.4, 3.7))
    for lab in ["MLP+aug", "Transformer+aug", "Equivariant"]:
        ms = [res[lab][a] for a in ag]
        plt.errorbar(ag, [m[0] for m in ms], yerr=[m[1] for m in ms], marker="o",
                     capsize=2, color=colors[lab], label=lab, linewidth=1.6)
    for lab, style in [("Bartlett-nominal", "--"), ("Bartlett-oracle", ":")]:
        plt.plot(ag, [res[lab][a][0] for a in ag], style, color=colors[lab], label=lab, linewidth=1.6)
    plt.yscale("symlog", linthresh=2)
    plt.xlabel("Coupling strength alpha"); plt.ylabel("OOD azimuth error (deg)")
    plt.title("DOA under unknown circulant mutual coupling (N=2000)")
    plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fig_coupling.png"), dpi=160); plt.close()
    print("Saved fig_coupling.png")


def print_pgf(out):
    res = out["results"]; ag = out["config"]["alpha_grid"]
    print("\n==== COUPLING SWEEP (alpha, OOD mean, OOD std) ====")
    for lab in ["MLP+aug", "Transformer+aug", "Equivariant"]:
        print(f"{lab}: " + " ".join(f"({a},{res[lab][a][0]:.2f})+-(0,{res[lab][a][1]:.2f})" for a in ag))
    for lab in ["Bartlett-nominal", "Bartlett-oracle"]:
        print(f"{lab}: " + " ".join(f"({a},{res[lab][a][0]:.2f})" for a in ag))


if __name__ == "__main__":
    main()