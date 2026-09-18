import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qsq.figures import configure_plots, save_figure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/noise.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def plot_aggregate(errors, noise_levels, dimensions, ylabel, output):
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for noise_index, sigma in enumerate(noise_levels[::-1]):
        samples = errors[len(noise_levels) - 1 - noise_index]
        lower, median, upper = np.percentile(samples, [25, 50, 75], axis=0)
        line = axis.plot(
            dimensions,
            median,
            label=rf"$\sigma=10^{{{int(np.log10(sigma))}}}$",
        )[0]
        axis.fill_between(
            dimensions, lower, upper, color=line.get_color(), alpha=0.18
        )
    axis.set_yscale("log")
    axis.set_xlabel("Krylov dimension")
    axis.set_ylabel(ylabel)
    axis.legend(loc="lower left", fontsize=11, ncol=1)
    save_figure(figure, output)


def main():
    args = parse_args()
    configure_plots()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        plot_aggregate(
            data["gibbs_errors"],
            data["noise_levels"],
            data["gibbs_dimensions"],
            "relative error",
            args.output_dir / "gibbs_w_noise.pdf",
        )
        plot_aggregate(
            data["monomial_errors"],
            data["noise_levels"],
            data["monomial_dimensions"],
            "relative error",
            args.output_dir / "monomial_w_noise.pdf",
        )


if __name__ == "__main__":
    main()
