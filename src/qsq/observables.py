import numpy as np


def gibbs_from_rule(nodes, weights, beta, time_step):
    # node = exp(-i E time_step), so angle(node)/time_step = -E; hence exp(beta*angle(node)/time_step) = exp(-beta E).
    # Relies on E*time_step staying within the principal branch (-pi, pi], true for this model.
    phases = np.angle(nodes)
    return weights @ np.exp(beta * phases / time_step)


def greens_function_from_rule(nodes, weights, frequencies, broadening, time_step):
    # node = exp(-i E time_step), so -angle(node)/time_step = E.
    # Relies on E*time_step staying within the principal branch (-pi, pi], true for this model.
    energies = -np.angle(nodes) / time_step
    poles = np.asarray(frequencies) + 1j * broadening
    return weights @ (1.0 / (energies[:, None] - poles[None, :]))