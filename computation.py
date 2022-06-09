from networkParameters import *
import numpy as np
import matplotlib.pyplot as plt


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

def compute_output_weights(trajectories):
    X = compute_state_matrix(trajectories)

    P = np.array(2 * np.sin(t / 20))
    P = P[:washout_time]
    regularisation_constant_readouts = 0.01
    output_weights_comp = ridge_regression(X, P, regularisation_constant_readouts)

    plt.plot(output_weights_comp, '.')
    plt.title("computed readout weights")
    plt.xlabel("neuron index")
    plt.ylabel("computed value")
    plt.show()
    return output_weights_comp

def compute_loading_weights(trajectories):
    regularisation_constant_internal = 0.1

    state_matrix_delayed = []
    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])
        state_matrix_delayed.append(trajectories[washout_time-1:-1, i])

    X = np.array(state_matrix)
    X_tilde = np.array(state_matrix_delayed)

    goal = np.arctan(X)

    return ridge_regression(X_tilde, goal, regularisation_constant_internal)

def compute_conceptor(trajectories, aperture):

    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])
    X = np.array(state_matrix)
    # R = np.dot(X, X.transpose()) / N
    R = np.corrcoef(X)
    C_0 = aperture ** -2 * np.identity(N)
    C = np.dot(R, np.linalg.inv(np.add(R, C_0)))
    # print(C)
    return C

def compute_conceptor_quota(C):
    return C.trace()/N

