# Quantum Szegő quadrature numerics

Reproducible numerical calculations for [Quantum Krylov Algorithm for Szegő Quadrature](https://arxiv.org/abs/2509.19195).

## Setup

Python 3.10 or newer is required. Install the package and its dependencies from this directory:

```bash
python3 -m pip install -e .
```

## Generate the figures

Generate the noiseless figures:

```bash
python3 scripts/random_laurent_poly.py
python3 scripts/gibbs_varying_beta.py
python3 scripts/gibbs_comparison.py
python3 scripts/greens_function.py
python3 scripts/greens_function_error.py
```

`greens_function.py` also writes bare, narrow-window (omega in [-28, -8]) real and imaginary panels (`re_gf_narrow_d_{d}.pdf`, `im_gf_narrow_d_{d}.pdf`) for insetting into the Green's function figure.

Generate the shared noisy dataset and both noisy figures:

```bash
python3 scripts/generate_noise_data.py
python3 scripts/plot_noise.py
```

The outputs are:

```text
figures/random_laurent_poly.pdf
figures/monomial_w_noise.pdf
figures/gibbs_varying_beta.pdf
figures/gibbs_comparison.pdf
figures/GF_figure_4x3.pdf
figures/gf_error.pdf
figures/gibbs_w_noise.pdf
```

## Numerical setup

The calculations use the 4x3 XYZ model with couplings `jx = jy = h = 1` and `jz = 2`, together with the half-filled checkerboard initial state. The Hamiltonian is diagonalized in its conserved half-filling sector. The resulting spectral measure is cached in `data/spectral_measure_3x4.npz` and reused by every calculation.

The noisy calculations use circular complex Gaussian noise,

```text
z = X + iY,    X,Y ~ N(0, sigma^2/2),
```

so that `E[|z|^2] = sigma^2`. They use 100 seeded realizations and noise RMS values from `1e-10` through `1e-2`. The degree-5 monomial calculation runs through Krylov dimension 20, while the `beta=1` Gibbs-state calculation runs through dimension 40. Both plots show the median and interquartile range. The generated numerical results are cached in `data/noise.npz`.
