import itertools

from tqdm import tqdm

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def get_basis(n_qubits):
    basis = {}
    for q, state in enumerate(itertools.product((0, 1), repeat = n_qubits)):
        basis[q] = state

    reverse_basis = {v: k for k, v in basis.items()}

    return basis, reverse_basis


class NoPair(Exception):
    pass


def find_pair(q_1, q_2, basis):
    state_1, state_2 = basis[q_1], basis[q_2]

    diffs = 0
    pair = []
    for position, (i, j) in enumerate(zip(state_1, state_2)):
        if i != j:
            pair.append(position)
            diffs += 1
            if diffs > 2:
                break

    if diffs == 2 and state_1[pair[0]] != state_1[pair[1]]:
        return pair

    raise NoPair


def construct_hamiltonian(basis, k = 1, a = 1):
    hamiltonian = sparse.dok_matrix((len(basis), len(basis)), dtype = np.complex128)

    for q_1 in basis:
        for q_2 in range(q_1 + 1):  # iterate over lower diagonal of matrix
            try:
                pair = find_pair(q_1, q_2, basis)
                distance = np.random.rand()  # TEMP! use pair to get distance!
                ham = np.sin(distance * k) / np.power(distance, a)
                hamiltonian[q_1, q_2] = hamiltonian[q_2, q_1] = ham
            except NoPair:
                pass

    return hamiltonian


def norm(psi):
    return np.vdot(psi, psi).real


def get_random_initial_state(basis):
    psi = np.random.rand(len(basis)) + 1j * np.random.rand(len(basis))
    return psi / np.sqrt(norm(psi))


def time_step__forward_euler(hamiltonian, psi, time_step):
    deriv = -1j * hamiltonian.dot(psi)

    return psi + (time_step * deriv)


if __name__ == "__main__":
    np.set_printoptions(linewidth = 200)

    basis, reverse_basis = get_basis(5)

    # print('Basis:')
    # for q, state in basis.items():
    #     print(f'{q:>2} -> {state}')
    #
    # print('\nReverse Basis:')
    # for state, q in reverse_basis.items():
    #     print(f'{state} -> {q:>2}')

    psi = get_random_initial_state(basis)
    # print(psi)
    # print(psi.shape)

    # try:
    #     print(find_pair(0, 1, basis))
    # except NoPair:
    #     print('yeah, no pair')
    # print(find_pair(1, 4, basis))

    ham = construct_hamiltonian(basis)
    # print(ham.toarray())

    times = np.linspace(0, 4, 50000)
    dt = np.abs(times[1] - times[0])
    states_over_time = [psi]

    for t in tqdm(times[1:]):
        psi = time_step__forward_euler(ham, psi, dt)
        states_over_time.append(psi)

    # for p in states_over_time:
    #     print(p)

    probabilities_over_time = [np.abs(state) ** 2 for state in states_over_time]
    # for n in probabilities_over_time:
    #     print(n)

    print(probabilities_over_time[-1])

    y_arrays = [[state[q] for state in probabilities_over_time] for q in basis]
    norm_vs_time = [sum(y) for y in probabilities_over_time]
    for q, y in enumerate(y_arrays):
        plt.plot(y, label = f'q = {q}')
    plt.plot(norm_vs_time, label = 'norm', color = 'black')

    plt.show()
