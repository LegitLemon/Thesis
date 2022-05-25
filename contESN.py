import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

def leaky_esn_conceptor(state, t, tau, a_input, w_connec, w_input, w_output, conceptor):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    # input_driven = np.dot(w_input, 2*np.sin(t/20))
    # new_state_input = np.tanh(old_state_input)

    old_state_input = np.dot(w_connec, x)
    new_state_input = np.dot(conceptor, np.tanh(old_state_input))

    dxdt = (1/tau)*(decay_input + new_state_input)

    return dxdt

def leaky_esn(state, t, tau, a_input, w_connec, w_input, w_output):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    input_driven = np.dot(w_input, 2*np.sin(t/20))
    old_state_input = np.dot(w_connec, x)

    # noise = np.random.normal(0, 0.00001, size=N)

    new_state_input = np.tanh(input_driven + old_state_input)

    dxdt = (1/tau)*(decay_input + new_state_input)

    # y = np.dot(w_output, dxdt)

    # return np.append(dxdt, y)
    return dxdt


def ridge_regression(X, P, rho):
    W_1 = np.linalg.inv((np.dot(X, X.transpose()) + rho*np.identity(N)))
    W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
    return W_opt


def compute_output_weights(trajectories):
    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])

    X = np.array(state_matrix)
    print(X.shape)

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
    regularisation_constant_internal = 0.0001

    state_matrix_delayed = []
    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])
        state_matrix_delayed.append(trajectories[washout_time-1:-1, i])

    X = np.array(state_matrix)
    X_tilde = np.array(state_matrix_delayed)

    P = np.arctan(X)

    return ridge_regression(X_tilde, P, regularisation_constant_internal)

def compute_conceptor(trajectories):
    aperture = 100
    state_matrix = []
    for i in range(N):
        state_matrix.append(trajectories[washout_time:, i])
    X = np.array(state_matrix)
    R = np.dot(X, X.transpose()) / X.shape[0]

    C_0 = aperture ** -2 * np.identity(N)
    C = np.dot(R, np.linalg.inv(np.add(R, C_0)))
    return C

def plot_liquid_states(trajectories):
    for i in range(6):
        index = random.randint(0, N-1)
        plt.plot(t, trajectories[:, index], label=i)
    plt.plot(t, np.sin(t/10), label="input signal")
    plt.xlabel('time')
    plt.title(f'Neuron states, tau={tau}, a=${a_internal}, N=${N}')
    plt.ylabel('x(t)')
    plt.legend()
    plt.show()


def plot_output(trajectories, used_weights):
    # output1 = trajectories[:, N]
    output2 = []
    for state in trajectories:
        output_point = np.dot(used_weights, state)
        output2.append(output_point)

    # plt.plot(t, output1)
    plt.plot(t, output2, label="test")
    plt.plot(t, 2*np.sin(t/20), label="input signal")
    plt.xlabel('time')
    plt.title(f'System output, tau={tau}, a=${a_internal}, N=${N}')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()

washout_time = 1500

# amount of neurons
N = 20

# leaking rate and time constant same for all neurons
a_internal = 0.2
tau = 2

# generate initial conditions
initial_conditions_neurons = np.random.standard_normal(N)
# generate network settings
internal_weights = np.random.standard_normal(size=(N, N))
input_weights = np.random.normal(0, 1, size=(N))
output_weights = np.random.normal(0, 1, size=(N)).transpose()
leaking_matrix = np.identity(N)*a_internal


# scale internal weights
desired_spectral_radius = 0.2
eigenvalues = eigvals(internal_weights)
scaling_factor = max(abs(eigenvalues))
print(scaling_factor)
internal_weights *= desired_spectral_radius/scaling_factor

# Initialise timescale
sampling_frequency = 10
time = 300
t = np.linspace(0, time, num=time*sampling_frequency)

parameters_training = (tau, leaking_matrix, internal_weights, input_weights, output_weights)

y_training = odeint(leaky_esn, initial_conditions_neurons, t, parameters_training)
plot_liquid_states(y_training)
plot_output(y_training, output_weights)

# prepare the conceptor and load the patterns
output_weights_computed = compute_output_weights(y_training)
internal_weights_computed = compute_loading_weights(y_training)
conceptor = compute_conceptor(y_training)

# Test the training of the output weights
parameters_test_system = (tau, leaking_matrix, internal_weights, input_weights, output_weights_computed)
y_test_system_readout = odeint(leaky_esn, initial_conditions_neurons, t, parameters_test_system)
plot_output(y_test_system_readout, output_weights_computed)

# Test the loading
parameters_test_system = (tau, leaking_matrix, internal_weights, input_weights, output_weights_computed)
y_test_system_loading = odeint(leaky_esn, initial_conditions_neurons, t, parameters_test_system)
plot_output(y_test_system_loading, output_weights_computed)

# Test the conceptor retrieval
parameters_test_conceptor = (tau, leaking_matrix, internal_weights_computed, input_weights, output_weights_computed, conceptor)
y_test_conceptor = odeint(leaky_esn_conceptor, initial_conditions_neurons, t, parameters_test_conceptor)
plot_output(y_test_conceptor, output_weights_computed)




