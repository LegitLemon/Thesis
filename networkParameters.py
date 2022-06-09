import numpy as np

washout_time = 1500
N = 200
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
