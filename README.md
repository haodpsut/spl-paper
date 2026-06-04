# Geometric Equivariance for Orientation-Robust Direction Finding in NTN

Code and manuscript for the IEEE Signal Processing Letters submission
*Geometric Equivariance as an Inductive Bias for Orientation-Robust Direction
Finding in Non-Terrestrial Networks* (single author: Phuc Hao Do, Da Nang
Architecture University).

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
- **Two sources (beating the classical limit).** On a single-snapshot two-source
  task, the learned models resolve sources about twice as close as the beamformer
  (from ~15 deg vs ~30 deg). The equivariant model alone resolves the hardest
  separation (12 deg) and uses an order of magnitude fewer parameters than the
  augmented baseline, while its in- and out-of-distribution curves coincide.

## Repository layout

```
experiment/
  doa_equivariance.py   # single-source: MLP / Transformer / +aug / C_M-equivariant / Bartlett
                        #   studies: data efficiency (vs training size), SNR sweep; 5 seeds
  doa_resolution.py     # two-source single-snapshot resolution; equivariant deconvolver vs
                        #   MLP+aug vs Bartlett; resolution probability vs separation; 3 seeds
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
python doa_resolution.py       # two-source resolution   -> results/results_resolution.json
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

The two-source study is single-snapshot, equal-power, narrowband, on a planar UCA;
a multi-snapshot setting would admit a subspace baseline such as MUSIC. The
equivariant advantage over augmentation in the two-source case is confined to the
hardest separation and to parameter count. See the paper for the full discussion.