import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsq.figures import configure_plots, save_figure
from qsq.observables import greens_function_from_rule
from qsq.quadrature import paper_problem, quadrature_rule


def greens_function(rule, frequencies, broadening, time_step):
    return greens_function_from_rule(*rule, frequencies, broadening, time_step)


def save_narrow_inset(frequencies, exact, approximation, part, output):
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot(frequencies, part(exact), "--", linewidth=4)
    axis.plot(frequencies, part(approximation), linewidth=4)
    axis.set_xticks([])
    axis.set_yticks([])
    save_figure(figure, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/GF_figure_4x3.pdf"))
    parser.add_argument("--frequency-step", type=float, default=0.1)
    args = parser.parse_args()

    dimensions = [5, 10, 30]
    energies, probabilities, _, time_step, _, moment_values = paper_problem(max(dimensions))
    rules = {
        dimension: quadrature_rule(moment_values, dimension, threshold=1e-13)
        for dimension in range(1, max(dimensions) + 1)
    }
    broadening = 0.1

    def exact_greens_function(frequencies):
        return probabilities @ (
            1.0 / (energies[:, None] - frequencies[None, :] - 1j * broadening)
        )

    configure_plots()

    # Main figure over the wide frequency window.
    frequencies = np.arange(-60, 20, args.frequency_step)
    exact = exact_greens_function(frequencies)
    figure, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
    for row, dimension in enumerate(dimensions):
        approximation = greens_function(rules[dimension], frequencies, broadening, time_step)
        axes[row, 0].plot(frequencies, exact.real, "--", label="exact")
        axes[row, 0].plot(frequencies, approximation.real, label="QSQ approximation")
        axes[row, 1].plot(frequencies, exact.imag, "--")
        axes[row, 1].plot(frequencies, approximation.imag)
        axes[row, 0].set_ylabel(r"Re$[G^R(\omega)]$")
        axes[row, 1].set_ylabel(r"Im$[G^R(\omega)]$")
        axes[row, 0].set_title(rf"$d={dimension}$")
        axes[row, 1].set_title(rf"$d={dimension}$")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel(r"$\omega$")
    axes[-1, 1].set_xlabel(r"$\omega$")
    save_figure(figure, args.output)

    # Bare narrow-window panels for insetting into the main figure.
    narrow_frequencies = np.arange(-28, -8, args.frequency_step)
    narrow_exact = exact_greens_function(narrow_frequencies)
    for dimension in dimensions:
        approximation = greens_function(rules[dimension], narrow_frequencies, broadening, time_step)
        save_narrow_inset(
            narrow_frequencies, narrow_exact, approximation, np.real,
            args.output.parent / f"re_gf_narrow_d_{dimension}.pdf",
        )
        save_narrow_inset(
            narrow_frequencies, narrow_exact, approximation, np.imag,
            args.output.parent / f"im_gf_narrow_d_{dimension}.pdf",
        )


if __name__ == "__main__":
    main()
