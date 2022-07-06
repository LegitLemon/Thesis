import numpy as np


washout_time = 1500
N = 100
a_internal = 0.2
tau = 2
desired_spectral_radius = 0.4


# Initialise timescale
sampling_frequency = 10
time = 300
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
