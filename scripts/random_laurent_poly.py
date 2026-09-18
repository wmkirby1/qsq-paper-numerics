import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsq.figures import configure_plots, save_figure
from qsq.quadrature import ideal_isometric_arnoldi_rules, paper_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/random_laurent_poly.pdf"))
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--max-dimension", type=int, default=21)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    _, probabilities, _, _, exact_nodes, _ = paper_problem(args.max_dimension)
    unitary = np.diag(exact_nodes)
    initial_state = np.sqrt(probabilities).astype(complex)
    rules = ideal_isometric_arnoldi_rules(
        unitary, initial_state, args.max_dimension
    )
    rng = random.Random(args.seed)
    errors = np.empty((args.max_degree, args.repetitions, args.max_dimension))

    for degree in range(1, args.max_degree + 1):
        for repetition in range(args.repetitions):
            coefficients = np.array([rng.gauss(0, 1) for _ in range(2 * degree + 1)])
            powers = np.arange(-degree, degree + 1)
            exact = probabilities @ ((exact_nodes[:, None] ** powers) @ coefficients)
            for dimension, (nodes, weights) in rules.items():
                approximation = weights @ ((nodes[:, None] ** powers) @ coefficients)
                errors[degree - 1, repetition, dimension - 1] = abs(approximation - exact) / abs(exact)

    configure_plots()
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for degree in range(1, args.max_degree + 1):
        axis.plot(
            range(1, args.max_dimension + 1),
            np.mean(errors[degree - 1], axis=0),
            marker=".",
            label=f"degree {degree}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Krylov dimension")
    axis.set_ylabel("relative error")
    axis.legend(fontsize=9, ncol=2)
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
