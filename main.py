import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from dynamics import *
from plotting import *
from computation import *
from networkParameters import *
import matplotlib.pyplot as plt
import random

def plot_aperture_response(training_data, internal_weights_computed, P, leaking_matrix, output_weights_used):
    aperture_values = [1, 60, 200, 10000]

    projection_data_output = []
    negation_data_output = []
    quotes = []
    for alpha in aperture_values:
        conceptor = compute_conceptor(training_data, alpha)
        quotes.append(compute_conceptor_quota(conceptor))
        # Test conceptor retrieval, use conceptor in update loop.
        parameters_test_conceptor = (tau, leaking_matrix, internal_weights_computed, conceptor, P)

        projection_data = odeint(leaky_esn_conceptor_projection, np.random.standard_normal(N), t, parameters_test_conceptor)
        negation_data = odeint(leaky_esn_conceptor_negation, np.random.standard_normal(N), t, parameters_test_conceptor)

        projection_point = []
        negation_point = []

        for step_number in range(len(projection_data)):
            projection_point.append(np.dot(output_weights_used, projection_data[step_number]))
            negation_point.append(np.dot(output_weights_used, negation_data[step_number]))

        projection_data_output.append(projection_point)
        negation_data_output.append(negation_point)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # for idx, alpha in enumerate(projection_data_output):
    #     axs[idx][right].set_title(f"alpha= {aperture_values[idx]} quota = {quotes[idx]}")
    #     axs[idx][right].plot(t, projection_data_output[idx], label="projection")
    #     axs[idx][right].plot(t, negation_data_output[idx], label="negation")
    #     axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    idx = 0
    right = 0
    axs[idx][right].set_title(f"alpha= {aperture_values[idx]} quota = {quotes[idx]}")
    axs[idx][right].plot(t, projection_data_output[idx], label="projection")
    axs[idx][right].plot(t, negation_data_output[idx], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    right = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[1]} quota = {quotes[1]}")
    axs[idx][right].plot(t, projection_data_output[idx], label="projection")
    axs[idx][right].plot(t, negation_data_output[idx], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    right = 0
    idx = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[2]} quota = {quotes[2]}")
    axs[idx][right].plot(t, projection_data_output[idx], label="projection")
    axs[idx][right].plot(t, negation_data_output[idx], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    right = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[3]} quota = {quotes[3]}")
    axs[idx][right].plot(t, projection_data_output[idx], label="projection")
    axs[idx][right].plot(t, negation_data_output[idx], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")


    print(quotes)
    plt.legend()
    plt.show()

def main():
    # generate initial conditions
    # generate network settings
    internal_weights = np.random.standard_normal(size=(N, N))
    input_weights = np.random.normal(0, 1, size=(N))
    output_weights = np.random.normal(0, 1, size=(N)).transpose()
    leaking_matrix = np.identity(N)*a_internal
    bias_vector = np.random.normal(0, 1, size=(N))

    # scale internal weights
    eigenvalues = eigvals(internal_weights)
    scaling_factor = max(abs(eigenvalues))
    print(scaling_factor)
    internal_weights *= desired_spectral_radius/scaling_factor


    parameters_training = (tau, leaking_matrix, internal_weights, input_weights, output_weights, bias_vector)

    y_training = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_training)

    Q, R = np.linalg.qr(compute_state_matrix(y_training))

    state_matrix_rank = np.linalg.matrix_rank(compute_state_matrix(y_training))

    # extract basis for X
    Q = Q[:, :state_matrix_rank]
    print(Q.shape)

    term0 = np.dot(Q.transpose(), Q)
    term1 = np.linalg.inv(term0)

    print(term1.shape)

    term2 = np.dot(Q, np.dot(term1, Q.transpose()))
    print(term2.shape)

    P = np.identity(N)-term2


    print("state matrix rank: ", state_matrix_rank)
    print("Q:", np.linalg.matrix_rank(Q))

    plot_liquid_states(y_training)
    plot_output(y_training, output_weights)

    # prepare the conceptor and load the patterns
    output_weights_computed = compute_output_weights(y_training)
    internal_weights_computed = compute_loading_weights(y_training)

    desired_spectral_radius_internal_weights = 1.2
    eigenvalues_internal_matrix = eigvals(internal_weights_computed)
    scaling_factor_internal_weights = max(abs(eigenvalues))
    print(scaling_factor_internal_weights)
    # internal_weights_computed *= desired_spectral_radius_internal_weights/scaling_factor_internal_weights

    aperture=60
    conceptor = compute_conceptor(y_training, aperture)

    # Test the training of the output weights
    parameters_test_system = (tau, leaking_matrix, internal_weights, input_weights, output_weights_computed, bias_vector)
    y_test_system_readout = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_test_system)
    plot_output(y_test_system_readout, output_weights_computed)

    # Test the loading. Use loaded reservoir and signal input, no conceptor
    parameters_test_system = (tau, leaking_matrix, internal_weights_computed, input_weights, output_weights_computed, bias_vector)
    y_test_system_loading = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_test_system)
    plot_liquid_states(y_test_system_loading)
    plot_output(y_test_system_loading, output_weights_computed)

    # Test conceptor retrieval, use conceptor in update loop.
    parameters_test_conceptor = (tau, leaking_matrix, internal_weights_computed, conceptor, P)
    y_test_conceptor_projection = odeint(leaky_esn_conceptor_projection, np.random.standard_normal(N), t, parameters_test_conceptor)
    y_test_conceptor_negation = odeint(leaky_esn_conceptor_negation, np.random.standard_normal(N), t, parameters_test_conceptor)

    plot_liquid_states(y_test_conceptor_projection)
    plot_control_errors(6)
    plot_output_retrieval(y_test_conceptor_projection, y_test_conceptor_negation, output_weights_computed)

    plot_aperture_response(y_training, internal_weights_computed, P, leaking_matrix, output_weights_computed)

if __name__ == "__main__":
    main()


