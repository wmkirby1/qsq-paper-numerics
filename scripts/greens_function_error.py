import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsq.figures import configure_plots, save_figure
from qsq.observables import greens_function_from_rule
from qsq.quadrature import paper_problem, quadrature_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/gf_error.pdf"))
    parser.add_argument("--max-dimension", type=int, default=197)
    parser.add_argument("--frequency-step", type=float, default=0.1)
    args = parser.parse_args()

    energies, probabilities, _, time_step, _, moment_values = paper_problem(args.max_dimension)
    frequencies = np.arange(-60, 20, args.frequency_step)
    broadening = 0.1
    exact = probabilities @ (
        1.0 / (energies[:, None] - frequencies[None, :] - 1j * broadening)
    )
    errors = []
    for dimension in range(1, args.max_dimension + 1):
        nodes, weights = quadrature_rule(moment_values, dimension, 1e-13)
        approximation = greens_function_from_rule(
            nodes, weights, frequencies, broadening, time_step
        )
        errors.append(np.mean(np.abs(approximation - exact)))

    dimensions = np.arange(1, args.max_dimension + 1)
    fit = np.polyfit(
        np.log(dimensions[:-1]),
        np.log(np.asarray(errors)[:-1]),
        1,
        w=np.sqrt(np.diff(np.log(dimensions))),
    )
    configure_plots()
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot(dimensions, errors, label="QSQ approximation error")
    axis.plot(
        dimensions,
        np.exp(np.polyval(fit, np.log(dimensions))),
        "--",
        label=rf"$O(d^{{{fit[0]:.2f}}})$",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$d$")
    axis.set_ylabel(r"error in $G^R(\omega)$")
    axis.legend()
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
