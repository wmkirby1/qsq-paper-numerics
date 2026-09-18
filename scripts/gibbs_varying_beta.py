import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsq.figures import configure_plots, save_figure
from qsq.observables import gibbs_from_rule
from qsq.quadrature import paper_problem, quadrature_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/gibbs_varying_beta.pdf"))
    parser.add_argument("--max-dimension", type=int, default=25)
    args = parser.parse_args()

    energies, probabilities, _, time_step, _, moment_values = paper_problem(args.max_dimension)
    rules = {
        dimension: quadrature_rule(moment_values, dimension, threshold=1e-14)
        for dimension in range(1, args.max_dimension + 1)
    }

    configure_plots()
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for beta in [10.0, 1.0, 0.1, 0.01, 0.001]:
        exact = probabilities @ np.exp(-beta * energies)
        errors = [
            abs(gibbs_from_rule(*rules[d], beta, time_step) - exact) / abs(exact)
            for d in rules
        ]
        axis.plot(range(1, args.max_dimension + 1), errors, label=rf"$\beta={beta:g}$")
    axis.plot(range(4, 26), [1e4 * np.exp(-d) for d in range(4, 26)], "k--")
    axis.plot(range(1, 17), [1e-2 * np.exp(-2 * d) for d in range(1, 17)], "k--")
    axis.set_yscale("log")
    axis.set_xlabel("Krylov dimension")
    axis.set_ylabel("relative error")
    axis.legend(loc="lower left")
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
