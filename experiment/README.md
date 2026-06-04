# Rotation-equivariant DOA experiment

Backs the paper *Geometric Equivariance as an Inductive Bias for Orientation-Robust
Direction Finding in Non-Terrestrial Networks* (`../paper/main.tex`).

## What it does

On a uniform circular array (UCA), rotating the source azimuth by one inter-sensor
angle equals a cyclic shift of the sensor index (group `C_M`). The script compares
three ways of handling that symmetry, plus a classical reference:

- **MLP**, **Transformer** — symmetry-agnostic (ignore the symmetry).
- **MLP+aug**, **Transformer+aug** — same models trained with random cyclic-shift
  augmentation (augment for the symmetry).
- **EquivCircConv** — `C_M`-equivariant circular-convolution network (build it in).
- **Bartlett** — classical, non-learned beamformer reference.

Two studies, each over 5 seeds (mean and std):
- **Data efficiency**: OOD azimuth error vs training size `{200,500,1000,2000,5000}` at 10 dB.
- **SNR robustness**: OOD error vs SNR `{0,5,10,15,20}` dB at training size 1000.

Models train on azimuths in `[0, pi)`; evaluation is on held-out in-distribution
`[0, pi)` and out-of-distribution `[pi, 2pi)` azimuths.

## Run

```bash
python doa_equivariance.py
```

Requirements: `torch`, `numpy`, `matplotlib` (CPU is enough; ~20-40 min for the full
multi-seed sweep). Fixed seeds, reproducible. The console prints pgfplots-ready
coordinate strings used in the paper.

## Outputs (in `results/`)

- `results_full.json` — all mean/std numbers, params, Bartlett reference.
- `fig_data_efficiency.png`, `fig_snr.png` — quick matplotlib previews.

## Mutual-coupling robustness check (`doa_coupling.py`)

`doa_coupling.py` adds circulant mutual coupling `x = C a(phi) + noise`. Because a
UCA coupling matrix is circulant it commutes with the cyclic shift, so coupling
preserves the `C_M` symmetry and the equivariant model stays correctly specified.
The equivariant OOD error stays near 1.2 deg up to nearest-neighbour coupling 0.4,
matching the coupling-aware oracle beamformer. Note: reciprocal (physical) coupling
does NOT bias single-source DOA, so this is a robustness check, not a regime where
learning beats classical. Multi-source resolution is the place to probe the latter.

## Headline finding

Augmentation and built-in equivariance both reach the classical accuracy at large
data, but the equivariant model is more data efficient (1.88 deg vs 2.90 deg OOD at
N=200), far more stable across seeds (std ~0.01 deg vs the augmented Transformer's
>30 deg, including a non-converging seed), and uses the fewest parameters (10.8k).
The classical beamformer matches all learned models on this clean single-source task.
