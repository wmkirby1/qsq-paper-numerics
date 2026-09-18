import itertools
from pathlib import Path

import numpy as np
import scipy.linalg

from .model import paper_spectral_measure


def paper_problem(max_dimension, cache=Path("data/spectral_measure_3x4.npz")):
    energies, probabilities, norm = paper_spectral_measure(cache=cache)
    time_step = np.pi / norm
    exact_nodes = np.exp(-1j * energies * time_step)
    moment_values = moments_from_measure(exact_nodes, probabilities, max_dimension)
    return energies, probabilities, norm, time_step, exact_nodes, moment_values


def moments_from_measure(nodes, probabilities, max_power):
    powers = np.arange(max_power + 1)
    return probabilities @ nodes[:, None] ** powers[None, :]


def krylov_matrices(moment_values, dimension):
    if len(moment_values) <= dimension:
        raise ValueError(f"dimension {dimension} requires moments through power {dimension}")

    gram = np.eye(dimension, dtype=complex)
    for row, column in itertools.combinations(range(dimension), 2):
        gram[row, column] = moment_values[column - row]
        gram[column, row] = gram[row, column].conj()

    projected_unitary = np.empty((dimension, dimension), dtype=complex)
    for row in range(dimension):
        for column in range(dimension):
            power = column + 1 - row
            projected_unitary[row, column] = (
                moment_values[power] if power >= 0 else moment_values[-power].conj()
            )
    return projected_unitary, gram


def stabilize_and_project(projected_unitary, gram, threshold):
    eigenvalues, eigenvectors = scipy.linalg.eigh(gram)
    # Spectral-shift regularization from Eq. (27).
    if eigenvalues[0] < threshold:
        regularized = eigenvalues - eigenvalues[0] + threshold
    else:
        regularized = eigenvalues
    inverse_sqrt = np.diag(1.0 / np.sqrt(regularized))
    orthogonalized = (
        inverse_sqrt
        @ eigenvectors.conj().T
        @ projected_unitary
        @ eigenvectors
        @ inverse_sqrt
    )
    left, _, right_adjoint = scipy.linalg.svd(orthogonalized)
    nearest_unitary = left @ right_adjoint
    basis_transform = eigenvectors @ np.diag(np.sqrt(regularized))
    return nearest_unitary, basis_transform


def quadrature_rule(moment_values, dimension, threshold):
    projected_unitary, gram = krylov_matrices(moment_values, dimension)
    nearest_unitary, basis_transform = stabilize_and_project(
        projected_unitary, gram, threshold
    )
    nodes, eigenvectors = np.linalg.eig(nearest_unitary)
    weights = np.abs(basis_transform[0] @ eigenvectors) ** 2
    weights /= weights.sum()
    return nodes, weights


def ideal_isometric_arnoldi_rules(unitary, initial_state, max_dimension):
    """Construct noiseless rules by the ideal isometric Arnoldi procedure."""
    basis = [np.array(initial_state, dtype=complex, copy=True)]
    for index in range(max_dimension - 1):
        vector = unitary @ basis[index]
        for _ in range(2):
            for previous in basis:
                vector -= (previous.conj() @ vector) * previous
        vector /= np.linalg.norm(vector)
        basis.append(vector)

    basis = np.column_stack(basis)
    projection_cache = basis.conj().T @ unitary @ basis

    rules = {}
    for dimension in range(1, max_dimension + 1):
        projected_unitary = projection_cache[:dimension, :dimension].copy()
        projected_unitary[:, -1] /= np.linalg.norm(projected_unitary[:, -1])
        nodes, eigenvectors = np.linalg.eig(projected_unitary.T)
        weights = np.abs(eigenvectors[0]) ** 2
        rules[dimension] = nodes, weights
    return rules
