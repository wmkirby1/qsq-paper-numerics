import argparse
from pathlib import Path

import numpy as np

from qsq.observables import gibbs_from_rule
from qsq.quadrature import paper_problem, quadrature_rule


def add_noise_to_moments(exact_moments, sigma, rng):
    """Add circular complex Gaussian noise with E[|z|^2] = sigma^2."""
    noisy = np.array(exact_moments, dtype=complex, copy=True)
    scale = sigma / np.sqrt(2.0)
    noise = rng.normal(scale=scale, size=len(noisy) - 1) + 1j * rng.normal(
        scale=scale, size=len(noisy) - 1
    )
    noisy[1:] += noise
    return noisy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/noise.npz"))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--gibbs-max-dimension", type=int, default=40)
    parser.add_argument("--monomial-max-dimension", type=int, default=20)
    parser.add_argument("--monomial-degree", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()
    if args.monomial_max_dimension > args.gibbs_max_dimension:
        parser.error("monomial maximum dimension cannot exceed Gibbs maximum dimension")
    return args


def main():
    args = parse_args()

    energies, probabilities, _, time_step, exact_nodes, exact_moments = paper_problem(
        args.gibbs_max_dimension
    )
    exact_gibbs = probabilities @ np.exp(-args.beta * energies)
    exact_monomial = probabilities @ exact_nodes**args.monomial_degree
    noise_levels = 10.0 ** np.arange(-10, -1)
    gibbs_errors = np.empty(
        (len(noise_levels), args.repetitions, args.gibbs_max_dimension)
    )
    monomial_errors = np.empty(
        (len(noise_levels), args.repetitions, args.monomial_max_dimension)
    )
    streams = iter(
        np.random.SeedSequence(args.seed).spawn(
            len(noise_levels) * args.repetitions
        )
    )

    for noise_index, sigma in enumerate(noise_levels):
        for repetition in range(args.repetitions):
            noisy_moments = add_noise_to_moments(
                exact_moments, sigma, np.random.default_rng(next(streams))
            )
            for dimension in range(1, args.gibbs_max_dimension + 1):
                nodes, weights = quadrature_rule(
                    noisy_moments, dimension, threshold=sigma
                )
                gibbs = gibbs_from_rule(nodes, weights, args.beta, time_step)
                gibbs_errors[noise_index, repetition, dimension - 1] = (
                    abs(gibbs - exact_gibbs) / abs(exact_gibbs)
                )
                if dimension <= args.monomial_max_dimension:
                    monomial = weights @ nodes**args.monomial_degree
                    monomial_errors[noise_index, repetition, dimension - 1] = (
                        abs(monomial - exact_monomial) / abs(exact_monomial)
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gibbs_errors=gibbs_errors,
        monomial_errors=monomial_errors,
        noise_model="circular-complex-gaussian",
        noise_levels=noise_levels,
        gibbs_dimensions=np.arange(1, args.gibbs_max_dimension + 1),
        monomial_dimensions=np.arange(1, args.monomial_max_dimension + 1),
        repetitions=args.repetitions,
        seed=args.seed,
        beta=args.beta,
        monomial_degree=args.monomial_degree,
    )


if __name__ == "__main__":
    main()
