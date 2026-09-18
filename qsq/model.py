from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse.linalg
from qiskit.quantum_info import SparsePauliOp


def xyz_hamiltonian(rows=3, columns=4, jx=1.0, jy=1.0, jz=2.0, field=1.0):
    graph = nx.grid_graph((rows, columns))
    graph = nx.relabel_nodes(graph, {node: index for index, node in enumerate(graph.nodes)})
    num_qubits = rows * columns

    hamiltonian = SparsePauliOp.from_sparse_list(
        [("Z", [qubit], field) for qubit in graph.nodes],
        num_qubits=num_qubits,
    )
    hamiltonian += SparsePauliOp.from_sparse_list(
        [("XX", edge, jx) for edge in graph.edges], num_qubits=num_qubits
    )
    hamiltonian += SparsePauliOp.from_sparse_list(
        [("YY", edge, jy) for edge in graph.edges], num_qubits=num_qubits
    )
    hamiltonian += SparsePauliOp.from_sparse_list(
        [("ZZ", edge, jz) for edge in graph.edges], num_qubits=num_qubits
    )
    return hamiltonian


def checkerboard_index(rows=3, columns=4):
    excitations = [
        rows * column + row
        for column in range(columns)
        for row in range(rows)
        if row % 2 == column % 2
    ]
    return sum(2**qubit for qubit in excitations)


def paper_spectral_measure(cache=None, rows=3, columns=4):
    if cache is not None and Path(cache).exists():
        with np.load(cache) as data:
            return data["energies"], data["probabilities"], float(data["norm"])

    hamiltonian = xyz_hamiltonian(rows=rows, columns=columns).to_matrix(sparse=True)
    particle_number = (rows * columns) // 2
    basis = np.array(
        [index for index in range(2 ** (rows * columns)) if index.bit_count() == particle_number]
    )
    sector_hamiltonian = hamiltonian[basis][:, basis].toarray()
    energies, eigenvectors = np.linalg.eigh(sector_hamiltonian)
    initial_index = np.flatnonzero(basis == checkerboard_index(rows, columns))[0]
    probabilities = np.abs(eigenvectors[initial_index]) ** 2
    norm = scipy.sparse.linalg.norm(hamiltonian, ord=2)

    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, energies=energies, probabilities=probabilities, norm=norm
        )
    return energies, probabilities, norm
