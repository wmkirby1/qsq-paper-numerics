import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from qsq.figures import configure_plots, save_figure
from qsq.observables import gibbs_from_rule
from qsq.quadrature import paper_problem, quadrature_rule


def fourier_approximation(energies, beta, time_step, degree):
    length = np.pi
    beta_rescaled = beta / time_step
    padding = 0.01 / degree
    padded_length = (1.0 + padding) * length
    result = np.zeros_like(energies, dtype=complex)
    phases = energies * time_step
    for order in range(-degree, degree + 1):
        denominator = beta_rescaled + 1j * order * np.pi / padded_length
        numerator = np.exp(denominator * padded_length) - np.exp(-denominator * padded_length)
        coefficient = numerator / (2.0 * padded_length * denominator)
        result += coefficient * np.exp(1j * order * phases)
    return result.real


def fixed_laurent_bound(degree, norm, beta):
    objective = lambda gamma: 4.0 * np.exp(
        beta * norm * (1.0 / gamma - 1.0) - (1.0 - gamma) * degree / 2.0
    )
    return scipy.optimize.minimize_scalar(
        objective, bounds=(1e-8, 1.0), method="bounded"
    ).fun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/gibbs_comparison.pdf"))
    parser.add_argument("--max-qsq-dimension", type=int, default=20)
    args = parser.parse_args()

    beta = 1.0
    laurent_degrees = np.array(sorted(set(int(1.3**power) for power in range(15))))
    fourier_degrees = np.array(sorted(set(int(1.3**power) for power in range(40))))
    max_degree = max(laurent_degrees.max(), args.max_qsq_dimension)
    energies, probabilities, norm, time_step, exact_nodes, moment_values = paper_problem(max_degree)
    exact_values = np.exp(-beta * energies)
    exact = probabilities @ exact_values

    qsq_errors = []
    for dimension in range(1, args.max_qsq_dimension + 1):
        nodes, weights = quadrature_rule(moment_values, dimension, 1e-14)
        approximation = gibbs_from_rule(nodes, weights, beta, time_step)
        qsq_errors.append(abs(approximation - exact) / abs(exact))

    optimal_errors = []
    for degree in laurent_degrees:
        powers = np.arange(-degree, degree + 1)
        design = exact_nodes[:, None] ** powers
        coefficients = np.linalg.lstsq(design, exact_values, rcond=None)[0]
        approximation = probabilities @ (design @ coefficients)
        optimal_errors.append(abs(approximation - exact) / abs(exact))

    fourier_errors = []
    for degree in fourier_degrees:
        approximation = probabilities @ fourier_approximation(
            energies, beta, time_step, degree
        )
        fourier_errors.append(abs(approximation - exact) / abs(exact))

    bound_degrees = np.arange(1, 1000, 50)
    bounds = [fixed_laurent_bound(degree, norm, beta) for degree in bound_degrees]

    configure_plots()
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot(fourier_degrees, fourier_errors, ":", label="Fourier")
    axis.plot(bound_degrees, bounds, "-.", label="fixed Laurent")
    axis.plot(laurent_degrees, optimal_errors, "--", label="optimal Laurent")
    axis.plot(range(1, args.max_qsq_dimension + 1), qsq_errors, label="QSQ")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(1e-16, 1e2)
    axis.set_xlabel("Krylov dimension")
    axis.set_ylabel("relative error")
    axis.legend()
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
