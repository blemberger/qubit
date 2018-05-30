import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import time
from tqdm import tqdm






def distance(pos1, pos2):
    dim = len(pos1)
    result = 0
    for d in range(0, dim):
        result += np.power( pos1[d] - pos2[d], 2)
    if result < np.power(.05, 2):
        print('very close qubits!')
    return np.sqrt(result)

def norm(state_real, state_imag):
    result = 0
    for q in range(0, len(state_real)):
        result += state_real[q]**2 + state_imag[q]**2
    return np.sqrt(result)

def normalize(state_real, state_imag):
    normed_real = np.zeros(len(state_imag))
    normed_imag = np.zeros(len(state_imag))
    n = norm(state_real, state_imag)
    for i in range(0, len(normed_imag)):
        normed_real[i] = state_real[i]/n
        normed_imag[i] = state_imag[i]/n
    return normed_real, normed_imag

def check_symmetric(matrix):
    result = 1
    for i in range(0, len(matrix[:,0])):
        for j in range(0, len(matrix[0,:])):
            if matrix[i,j] != matrix[j,i]:
                result = 0
    return result




def lattice(dim, side):
    result = np.zeros([np.power(side, dim), dim])
    for i in range(1, np.power(side, dim)):
        new_coords = np.zeros([dim])
        for d in range(0, dim):
            new_coords[d] = result[i-1, d]
        first_nonmax_column = 0
        for d in range(0, dim):
            if result[i-1,d] != (side-1):
                first_nonmax_column = d
                break
        new_coords[first_nonmax_column] += 1
        for d in range(0, dim):
            if d<first_nonmax_column:
                new_coords[d] = 0
        result[i, :] = new_coords
    return result

def random_sample_geometry(dim, qubits):
    result = np.zeros([qubits, dim])
    for i in range(0, qubits):
        for d in range(0, dim):
            result[i, d] = np.random.rand()
    return result

def generate_H(basis, lattice, a, scale, k):
    dim = np.power(2, len(lattice[:,0]) )
    result_real, result_imag = np.zeros([dim, dim]), np.zeros([dim, dim])
    for i in range(0, dim):
        for j in range(0, dim):
            qubit_pairs = find_pair(basis[i], basis[j])
            if qubit_pairs != [0,0]:
                rij = distance(lattice[qubit_pairs[0]], lattice[qubit_pairs[1]])
                result_real[i,j] = np.sin(rij*k) * np.power(scale * rij, -a)
                result_imag[i,j] = 0
    return result_real, result_imag

def find_pair(q1, q2):
    diff_count = 0
    pair = [0,0]
    for i in range(0, len(q1)):
        if q1[i] != q2[i]:
            if diff_count == 2:
                pair = [0,0]
                break
            else:
                pair[diff_count] = i
                diff_count += 1
    return pair

def generate_basis(qubits):
    basis_length = np.power(2, qubits)
    result = np.zeros([basis_length, qubits])
    for q in range(0, basis_length):
        binary_q = "{0:b}".format(q)
        for i in range(0, qubits):
            if i < len(binary_q):
                result[q, i] = binary_q[len(binary_q)-i-1]
    return result




def time_step(step, initial_state_real, initial_state_imag, H_real, H_imag):
    final_state_real = initial_state_real.copy()
    final_state_imag = initial_state_imag.copy()
    for q1 in range(0, Hdim):
        c_change_real, c_change_imag = 0, 0
        for q2 in range(0, Hdim):
            c_change_real += H_real[q1, q2] * initial_state_imag[q2] + H_imag[q1, q2] * initial_state_real[q2]
            c_change_imag += -H_real[q1, q2] * initial_state_real[q2] + H_imag[q1, q2] * initial_state_imag[q2]
        final_state_real[q1] += step * c_change_real
        final_state_imag[q1] += step * c_change_imag
    return final_state_real, final_state_imag

def time_evolution(times, initial_state_real, initial_state_imag, H_real, H_imag):
    result = np.zeros([len(times), len(initial_state_imag), 2])
    time_count = 0
    for t in tqdm(times):
        if t == 0:
            for i in range(0, len(initial_state_imag)):
                result[time_count, i, 0] = initial_state_real[i]
                result[time_count, i, 1] = initial_state_imag[i]
        else:
            previous_state_real, previous_state_imag = result[time_count-1, :, 0], result[time_count-1, :, 1]
            new_state = time_step(times[1]-times[0], previous_state_real, previous_state_imag, H_real, H_imag )
            for i in range(0, len(initial_state_imag)):
                result[time_count, i, 0] = new_state[0][i]
                result[time_count, i, 1] = new_state[1][i]
        time_count += 1
    return result

def measure_qubit_z(basis, index, state_real, state_imag):
    result = 0
    for q in range(0, len(state_real)):
        if basis[q, index] == 1:
            result += state_real[q]**2 + state_imag[q]**2
    return result

def measurement_evolution(basis, index, time_evolution_output):
    result = np.zeros(len(time_evolution_output[:,0,0]))
    for t in range(0, len(result)):
        current_state_real = time_evolution_output[t, :, 0]
        current_state_imag = time_evolution_output[t, :, 1]
        result[t] = measure_qubit_z(basis, index, current_state_real, current_state_imag)
    return result

def norm_evolution(time_evolution_output):
    result = np.zeros(len(time_evolution_output[:,0,0]))
    for t in range(0, len(result)):
        for i in range(0, len(time_evolution_output[0,:,0])):
            result[t] += time_evolution_output[t, i, 0]**2 + time_evolution_output[t,i,1]**2
    return result

def rho_element_evolution(Basis, index, switched, row, column, time_evolution_output):
    result_real, result_imag = np.zeros([len(time_evolution_output[:,0,0])]), np.zeros([len(time_evolution_output[:,0,0])])
    for t in range(0, len(result_real)):
        current_state_real = time_evolution_output[t, :, 0].copy()
        current_state_imag = time_evolution_output[t, :, 1].copy()
        reduced_rho = reduced_rho_1(Basis, index, switched, current_state_real, current_state_imag)
        reduced_rho_real = reduced_rho[0]
        reduced_rho_imag = reduced_rho[1]
        result_real[t] = reduced_rho_real[row, column]
        result_imag[t] = reduced_rho_imag[row, column]
    return result_real, result_imag

def magnitude_evolution(list_real, list_imag):
    result = np.zeros([len(list_imag)])
    for t in range(0, len(list_imag)):
        result[t] = np.sqrt( list_imag[t]**2 + list_real[t]**2)
    return result



def generate_switched_pairs(basis):
    result = np.zeros([len(basis[:,0]), len(basis[0,:])])
    for q in tqdm(range(0, len(result[:,0]))):
        for i in range(0, len(result[0,:])):
            result[q, i] = find_switched_index(basis, q, i)
    return result

def find_switched_index(basis, element, index):
    switched_index = -1
    for q in range(0, len(basis[:,0])):
        test = 0
        for i in range(0, len(Basis[0,:])):
            if ((Basis[q,i] == Basis[element,i])and(i != index))or((Basis[q,i] != Basis[element,i])and(i == index)):
                test += 1
        if test == len(Basis[0,:]):
            if switched_index != -1:
                print('two matching basisss??')
            switched_index = q
    return switched_index



def generate_product_state_1(Basis, index, P_up, phase):
    P_down = 1-P_up
    state_real, state_imag = np.zeros([len(Basis[:,0])]), np.zeros([len(Basis[:,0])])
    rest_state_real, rest_state_imag = np.zeros([len(Basis[:,0])]), np.zeros([len(Basis[:,0])])
    for q in range(0, len(Basis[:,0])):
        if Basis[q,index] == 0:
            mag = np.random.rand()
            angle = np.random.rand() * 2*np.pi
            rest_state_real[q] = mag*np.cos(angle)
            rest_state_imag[q] = mag*np.sin(angle)
    normed = normalize(rest_state_real, rest_state_imag)
    rest_state_real = normed[0]
    rest_state_imag = normed[1]
    for q in range(0, len(Basis[:,0])):
        if Basis[q, index] == 0:
            if rest_state_imag[q] == 0:
                print('rut row')
                break
            state_real[q] = (rest_state_real[q]*np.cos(phase) - rest_state_imag[q]*np.sin(phase)) * np.sqrt(P_down)
            state_imag[q] = (rest_state_imag[q]*np.cos(phase) + rest_state_real[q]*np.sin(phase)) * np.sqrt(P_down)
        elif Basis[q, index] == 1:
            switched_basis_index = find_switched_index(Basis, q, index)
            state_real[q] = rest_state_real[switched_basis_index] * np.sqrt(P_up)
            state_imag[q] = rest_state_imag[switched_basis_index] * np.sqrt(P_up)
        else:
            print('yikes!')
    return normalize(state_real, state_imag)

def reduced_rho_1(Basis, index, switched, state_real, state_imag):
    result_real, result_imag = np.zeros([2,2]), np.zeros([2,2])
    for q in range(0, len(state_imag)):
        if Basis[q, index] == 1:
            result_real[1,1] += state_real[q]**2 + state_imag[q]**2
            switched_index = switched[q, index]
            result_real[0, 1] += state_real[q]*state_real[switched_index] + state_imag[q]*state_imag[switched_index]
            result_imag[0, 1] += -state_real[q]*state_imag[switched_index] + state_imag[q]*state_real[switched_index]
        elif Basis[q, index] == 0:
            result_real[0,0] += state_real[q]**2 + state_imag[q]**2
        else:
            print('basis with non-binary value')
    return result_real, result_imag






dim = 3
qubits = 5
size_scale = 1
k = 1
Hdim = np.power( 2, qubits )
a = 1
times = np.arange(0 , 4 , .00008)
# initial_state_real = np.ones(Hdim)/np.sqrt(Hdim)
# initial_state_imag = np.zeros(Hdim)



Lattice = random_sample_geometry(dim, qubits)
Basis = generate_basis(qubits)
Switched_Pairs = generate_switched_pairs(Basis)

initial_state = generate_product_state_1(Basis, 1, .4, np.pi/3)

real0 = np.sqrt(.4)*np.sqrt(1-.4)*np.cos(np.pi/3)
imag0 = np.sqrt(.4)*np.sqrt(1-.4)*np.sin(np.pi/3)

initial_state_real = initial_state[0]
initial_state_imag = initial_state[1]

Ham_real, Ham_imag = generate_H(Basis, Lattice, a, size_scale, k)
evolution = time_evolution(times, initial_state_real, initial_state_imag, Ham_real, Ham_imag )



rho10 = rho_element_evolution(Basis, 1, Switched_Pairs, 0, 1, evolution)
rho10_real, rho10_imag = rho10[0], rho10[1]
offdiag_evolution = magnitude_evolution(rho10_real, rho10_imag)

PZ = measurement_evolution(Basis, 1, evolution)

norm_t = norm_evolution(evolution)

plt.plot(times, offdiag_evolution)
plt.plot(times, PZ)
plt.plot(times, norm_t)
plt.show()