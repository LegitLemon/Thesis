import numpy as np

number_of_patterns = 3

washout_time = 20
N = 500
a_internal = 0.6
tau = 3
desired_spectral_radius = 0.4


# Initialise timescale
sampling_frequency = 10
time = 5
t = np.linspace(0, time, num=time*sampling_frequency)


control_errors_projection = []
conceptor_distance_projection = []

control_errors_negation = []
conceptor_distance_negation = []
conceptor_distance_no_control = []


# Q, R = np.linalg.qr(state_matrix)
# state_matrix_rank = np.linalg.matrix_rank(state_matrix)
# # extract basis for X
# Q = Q[:, :state_matrix_rank]
# print(Q.shape)
# term0 = np.dot(Q.transpose(), Q)
# term1 = np.linalg.inv(term0)
# print(term1.shape)
# term2 = np.dot(Q, np.dot(term1, Q.transpose()))
# print(term2.shape)
# P = np.identity(N)-term2
# print("state matrix rank: ", state_matrix_rank)
# print("Q:", np.linalg.matrix_rank(Q))
# plot_liquid_states(y_training, title="Neuron status during training")
# plot_output(y_training, output_weights, title="untrained output signal")

# # generate network settings
# internal_weights = np.random.standard_normal(size=(N, N))
# input_weights = np.random.normal(0, 1, size=(N))
# output_weights = np.random.normal(0, 1, size=(N)).transpose()
# leaking_matrix = np.identity(N)*a_internal
# bias_vector = np.random.normal(0, 1, size=(N))
#
# # scale internal weights
# eigenvalues = eigvals(internal_weights)
# scaling_factor = max(abs(eigenvalues))
# print(scaling_factor)
# internal_weights *= desired_spectral_radius/scaling_factor
#
# training_data = []
# for i in range(number_of_patterns):
#     parameters_training = (tau, leaking_matrix, internal_weights, input_weights, i)
#     y_training = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_training)
#     training_data.append(y_training)
#
#
# # prepare the conceptor and load the patterns
# output_weights_computed = compute_output_weights(training_data)
# plot_neuron_states(training_data)
#
# test_data_output = []
# for i in range(number_of_patterns):
#     parameters_training = (tau, leaking_matrix, internal_weights, input_weights, i)
#     y_training = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_training)
#     test_data_output.append(y_training)
#
# test_training_output_weights(test_data_output, output_weights_computed)
#
# regularisation_internal = 10
# internal_weights_computed = compute_loading_weights(training_data, regularisation_internal)
#
# aperture = 6000
# conceptors = compute_conceptors(training_data, aperture)
#
# # Test conceptor retrieval
# conceptor_retrieval(conceptors, leaking_matrix, internal_weights_computed, output_weights_computed)
# # plot_aperture_response(y_training, internal_weights_computed, P, leaking_matrix, output_weights_computed)
