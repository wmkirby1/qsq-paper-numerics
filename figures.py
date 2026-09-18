import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib.pyplot as plt


def configure_plots():
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )


def save_figure(figure, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
