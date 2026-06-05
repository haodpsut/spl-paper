# Geometric Equivariance for Orientation-Robust Direction Finding in NTN

Code and manuscript for the IEEE Signal Processing Letters submission
*Geometric Equivariance as an Inductive Bias for Orientation-Robust Direction
Finding in Non-Terrestrial Networks*.

Authors: **Do Phuc Hao** (Da Nang Architecture University) and **Truong Duy Dinh**
(Posts and Telecommunications Institute of Technology, corresponding author).

On a uniform circular array (UCA), rotating the source azimuth by one inter-sensor
angle equals a cyclic shift of the sensor index, the group `C_M`. We study how this
geometric symmetry should enter a learned direction-of-arrival (DOA) estimator:
ignore it, augment for it, or build it in with a `C_M`-equivariant
circular-convolution network.

## Key results

- **Single source (equivariance vs augmentation).** Both built-in equivariance and
  cyclic-shift augmentation reach the classical accuracy at large data, but
  equivariance is more data efficient, far more stable across seeds, and uses the
  fewest parameters. A classical Bartlett beamformer matches all learned models on
  this clean task.
- **Two sources (robustness to coherence).** On a multi-snapshot two-source task,
  MUSIC resolves uncorrelated sources best, but it collapses for coherent sources
  (rank-deficient signal covariance), where the equivariant model, deconvolving the
  coherence-robust conventional spectrum, still resolves down to about half the
  beamwidth (0.94 at 15 deg vs MUSIC 0.47). The learned model complements the
  subspace method rather than displacing it.

## Repository layout

```
experiment/
  doa_equivariance.py   # single-source: MLP / Transformer / +aug / C_M-equivariant / Bartlett
                        #   studies: data efficiency (vs training size), SNR sweep; 5 seeds
  doa_resolution_ms.py  # two-source, multi-snapshot: equivariant deconvolver vs MUSIC vs
                        #   Bartlett; uncorrelated and coherent sources; 3 seeds (PAPER)
  doa_resolution.py     # earlier single-snapshot variant (superseded by _ms)
  doa_coupling.py       # robustness under circulant mutual coupling (preserves C_M)
  results/              # results JSON + preview figures produced by the scripts
paper/
  main.tex              # the Letter (IEEEtran, pgfplots figures, self-contained bibliography)
  cover_letter.tex      # cover letter for IEEE SPL
```

## Reproduce

Requirements: `python>=3.10`, `torch`, `numpy`, `matplotlib` (CPU is enough).

```bash
cd experiment
python doa_equivariance.py     # single-source studies  -> results/results_full.json
python doa_resolution_ms.py    # two-source vs MUSIC      -> results/results_resolution_ms.json
python doa_coupling.py         # mutual-coupling robustness check
```

Each script uses fixed seeds and prints pgfplots-ready coordinates used in the
paper. The console output buffers when redirected to a file; the result JSON is
written at the end.

## Build the paper

```bash
cd paper
pdflatex main && pdflatex main
pdflatex cover_letter
```

## Notes and honest scope

The two-source study is multi-snapshot (T=10), equal-power, narrowband, on a planar
UCA, and compares against MUSIC. The two-source value is specifically robustness to
source coherence: MUSIC remains the method of choice for uncorrelated sources, and we
do not claim to displace it there. See the paper for the full discussion.

## Acknowledgment

This work has been sponsored by the scientific research from Posts and
Telecommunications Institute of Technology, Vietnam.