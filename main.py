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
    no_control_data_output = []
    quotes = []
    for alpha in aperture_values:
        conceptor = compute_conceptor(training_data, alpha)
        quotes.append(compute_conceptor_quota(conceptor))
        # Test conceptor retrieval, use conceptor in update loop.
        parameters_test_conceptor = (tau, leaking_matrix, internal_weights_computed, conceptor, P)

        projection_data = odeint(leaky_esn_conceptor_projection, np.random.standard_normal(N), t, parameters_test_conceptor)
        negation_data = odeint(leaky_esn_conceptor_negation, np.random.standard_normal(N), t, parameters_test_conceptor)
        no_control_data = odeint(leaky_esn_no_control, np.random.standard_normal(N), t, parameters_test_conceptor)

        projection_point = []
        negation_point = []
        no_control_point = []

        for step_number in range(len(projection_data)):
            projection_point.append(np.dot(output_weights_used, projection_data[step_number]))
            negation_point.append(np.dot(output_weights_used, negation_data[step_number]))
            no_control_point.append(np.dot(output_weights_used, no_control_data[step_number]))

        projection_data_output.append(projection_point)
        negation_data_output.append(negation_point)
        no_control_data_output.append(no_control_point)

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
    axs[idx][right].plot(t, no_control_data_output[idx], label="no control")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    right = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[1]} quota = {quotes[1]}")
    axs[idx][right].plot(t, projection_data_output[1], label="projection")
    axs[idx][right].plot(t, negation_data_output[1], label="negation")
    axs[idx][right].plot(t, no_control_data_output[1], label="no control")

    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")

    right = 0
    idx = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[2]} quota = {quotes[2]}")
    axs[idx][right].plot(t, projection_data_output[2], label="projection")
    axs[idx][right].plot(t, negation_data_output[2], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")
    axs[idx][right].plot(t, no_control_data_output[2], label="no control")

    right = 1
    axs[idx][right].set_title(f"alpha= {aperture_values[3]} quota = {quotes[3]}")
    axs[idx][right].plot(t, projection_data_output[3], label="projection")
    axs[idx][right].plot(t, negation_data_output[3], label="negation")
    axs[idx][right].plot(t, 2*np.sin(t/20), label="input signal")
    axs[idx][right].plot(t, no_control_data_output[3], label="no control")


    print(quotes)
    plt.legend()
    plt.show()

def sweep_loading(y_training, leaking_matrix, input_weights, output_weights_computed, bias_vector):
    loading_options = np.linspace(0.05, 0.1, num=5)
    for option in loading_options:
        print(option)
        internal_weights_computed = compute_loading_weights(y_training, option)
        parameters_test_system = (tau, leaking_matrix, internal_weights_computed, input_weights, output_weights_computed, bias_vector)
        y_test_system_loading = odeint(leaky_esn, np.random.standard_normal(N), t, parameters_test_system)
        plot_liquid_states(y_test_system_loading)
        plot_output(y_test_system_loading, output_weights_computed)


# Only do when one pattern is loaded
def test_loading(leaking_matrix, internal_weights_computed, output_weights_computed):
    # Test the loading. Use loaded reservoir and signal input, no conceptor
    parameters_test_system = (tau, leaking_matrix, internal_weights_computed)
    initial_condition_loading = state_matrix[:, 500]
    y_test_system_loading = odeint(test_loading, initial_condition_loading, t, parameters_test_system)
    plot_liquid_states(y_test_system_loading, f'Liquid states loading, reguliser {regularisation_internal}')
    plot_output(y_test_system_loading, output_weights_computed, f'loading testing output, reguliser {regularisation_internal}')


def conceptor_retrieval(conceptors, leaking_matrix, internal_weights_computed, output_weights):
    for i in range(number_of_patterns):
        conceptor = conceptors[i]
        parameters_test_conceptor = (tau, leaking_matrix, internal_weights_computed, conceptor)
        initial_condition = np.random.standard_normal(N)
        y_test_conceptor_negation = odeint(leaky_esn_conceptor_negation, initial_condition, t, parameters_test_conceptor)
        y_test_conceptor_no_control = odeint(leaky_esn_no_control, initial_condition, t, parameters_test_conceptor)

        plot_states_with_output(y_test_conceptor_negation, output_weights, get_input(i, t), "plot with control")
        plot_states_with_output(y_test_conceptor_no_control, output_weights, get_input(i, t), "plot without control")
    # plot_control_errors(6)


# Normalise data to be within 0,1
def normalise_time_series(data):
    min_val = min(data)
    max_val = max(data)
    for idx, val in enumerate(data):
        data[idx] = (val-min_val)/(max_val-min_val)
    return data

def couple_pairs(data, tau=17):
    x, y = [], []
    for idx, dat in enumerate(data):
        if idx > 17:
            x.append(dat)
            y.append(data[idx-tau])
    return x, y

def main():
    initial_condition = [0, 1, 1]
    parameters = (t)
    data_lorenz = odeint(lorenz, initial_condition, parameters)
    x_data = [x[0] for x in data_lorenz]
    z_data = [z[2] for z in data_lorenz]

    x_data = normalise_time_series(x_data)
    z_data = normalise_time_series(z_data)

    plt.plot(x_data, z_data)
    plt.show()

    initial_condition = [1, 1, 0]

    data_rossler = odeint(rossler, initial_condition, parameters)
    x_data = [x[0] for x in data_rossler]
    y_data = [y[1] for y in data_rossler]

    x_data = normalise_time_series(x_data)
    y_data = normalise_time_series(y_data)

    plt.plot(x_data, y_data)
    plt.show()

    mackey_glass_data = mackey_glass()
    x_data, x_delayed_data = couple_pairs(mackey_glass_data)
    x_data = normalise_time_series(x_data)
    x_delayed_data = normalise_time_series(x_delayed_data)

    plt.plot(x_data, x_delayed_data)
    plt.show()


if __name__ == "__main__":
    main()


