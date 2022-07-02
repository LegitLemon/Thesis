from networkParameters import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def ridge_regression(X, P, rho):
    W_1 = np.linalg.inv((np.dot(X, X.transpose()) + rho*np.identity(N)))
    W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
    return W_opt

def compute_state_matrix(trajectories):
    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])
    X = np.array(state_matrix)
    return X

def compute_delayed_state_matrix(trajectories):
    state_matrix_delayed = []
    for i in range(N):
        state_matrix_delayed.append(trajectories[washout_time-1:-1, i])
    X_tilde = np.array(state_matrix_delayed)
    return X_tilde


def compute_output_weights(trajectories):
    P = []
    X = []
    for i in range(number_of_patterns):
        current_state_matrix = compute_state_matrix(trajectories[i])
        current_pattern = get_input(i, t)[:washout_time]
        if i == 0:
            X = current_state_matrix
            P = current_pattern
        else:
            X = np.hstack((X, current_state_matrix))
            P = np.hstack((P, current_pattern))

    regularisation_constant_readouts = 0.01
    output_weights_comp = ridge_regression(X, P, regularisation_constant_readouts)

    plt.plot(output_weights_comp, '.')
    plt.title("computed readout weights")
    plt.xlabel("neuron index")
    plt.ylabel("computed value")
    plt.show()
    return output_weights_comp

def compute_loading_weights(trajectories, regularisation_constant_internal = 0.08):
    X = []
    X_tilde = []
    for i in range(number_of_patterns):
        current_state_matrix = compute_state_matrix(trajectories[i])
        current_delayed_state_matrix = compute_delayed_state_matrix(trajectories[i])
        if i == 0:
            X = current_state_matrix
            X_tilde = current_delayed_state_matrix
        else:
            X = np.hstack((X, current_state_matrix))
            X_tilde = np.hstack((X_tilde, current_delayed_state_matrix))

    goal = np.arctan(X)

    return ridge_regression(X_tilde, goal, regularisation_constant_internal)

def compute_conceptor(trajectories, aperture):
    X = compute_state_matrix(trajectories)
    R = np.corrcoef(X)
    C_0 = aperture ** -2 * np.identity(N)
    C = np.dot(R, np.linalg.inv(np.add(R, C_0)))
    return C

def compute_conceptors(trajectories, aperture):
    conceptors = []
    for i in range(number_of_patterns):
        conceptors.append(compute_conceptor(trajectories[i], aperture))
    return conceptors

def compute_conceptor_quota(C):
    return C.trace()/N




def generate_rossler_input():
    pass

def generate_mg_input():
    pass


# Normalise data to be within 0,1
def normalise_time_series(data):
    min_val = min(data)
    max_val = max(data)
    for idx, val in enumerate(data):
        data[idx] = (val-min_val)/(max_val-min_val)
    return data

