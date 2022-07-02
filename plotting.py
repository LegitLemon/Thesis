from networkParameters import *
import random
import matplotlib.pyplot as plt
from dynamics import *
path = "plots/"

def plot_control_errors(amount, title="Difference between conceptor projection and current state"):
    for i in range(amount):
        index = random.randint(0, N-1)
        print(index)
        data_projection = np.array(control_errors_projection)[:, index]
        data_negation = np.array(control_errors_negation)[:, index]
        plt.plot(data_projection, color="red", label="projection")
        plt.plot(data_negation, color="blue", label="negation")
    plt.ylabel("error")
    plt.xlabel("time")
    plt.legend()
    plt.title("Error plots evolving over time")
    plt.show()

    plt.plot(conceptor_distance_projection)
    plt.plot(conceptor_distance_negation)
    plt.title(title)
    plt.savefig(path+title)
    plt.show()


def plot_liquid_states(trajectories, title=f'Neuron states, tau={tau}, a=${a_internal}, N=${N}', output=2*np.sin(t/20)):
    for i in range(6):
        index = random.randint(0, N-1)
        plt.plot(t, trajectories[:, index], label=i)
    plt.plot(t, output, label="input signal")
    plt.xlabel('time')
    plt.title(title)
    plt.ylabel('x(t)')
    plt.legend()
    # plt.savefig(path+title)
    plt.show()


def plot_output(trajectories, used_weights, title=f'System output, tau={tau}, a=${a_internal}, N=${N}', output=2*np.sin(t/20)):
    # output1 = trajectories[:, N]
    output2 = []
    for state in trajectories:
        output_point = np.dot(used_weights, state)
        output2.append(output_point)

    # plt.plot(t, output1)
    plt.plot(t, output2, label="test")
    plt.plot(t, output, label="input signal")
    plt.xlabel('time')
    plt.title(title)
    plt.ylabel('y(t)')
    plt.legend()
    # plt.savefig(path+title)
    plt.show()

def plot_output_retrieval(trajectories_projection, trajectories_negation, trajectories_no_control, used_weights, title=f'System output, tau={tau}, a=${a_internal}, N=${N}'):
    output1 = []
    output2 = []
    output3 = []
    for state_number in range(len(trajectories_negation)):
        output_point_projection = np.dot(used_weights, trajectories_projection[state_number])
        output1.append(output_point_projection)

        output_point_negation = np.dot(used_weights, trajectories_negation[state_number])
        output2.append(output_point_negation)

        output_point_no_control = np.dot(used_weights, trajectories_no_control[state_number])
        output3.append(output_point_no_control)


    # plt.plot(t, output1)
    plt.plot(t, output1, label="projection retrieval")
    plt.plot(t, output2, label="negation retrieval")
    plt.plot(t, output3, label="no control")
    plt.plot(t, 2*np.sin(t/20), label="goal signal")
    plt.xlabel('time')
    plt.title(title)
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig(path+title)
    plt.show()


def plot_neuron_states(trajectories):
    for i in range(number_of_patterns):
        plot_liquid_states(trajectories[i], output=get_input(i, t))

def test_training_output_weights(trajectories, used_weights):
    for i in range(number_of_patterns):
        plot_output(trajectories[i], used_weights, output=get_input(i, t))


def plot_states_with_output(trajectories, used_weights, input, title):
    output = []
    for state in trajectories:
        output_point = np.dot(used_weights, state)
        output.append(output_point)

    fig, axs = plt.subplots(2, sharex=True, sharey=False)
    axs[0].plot(t, output, label="Conceptor retrieval")
    axs[0].plot(t, input, label="Goal signal")

    for i in range(6):
        index = random.randint(0, N-1)
        axs[1].plot(t, trajectories[:, index], label=i)

    plt.title(title)
    plt.show()

def plot_input_data_attractors(input_data, title="Input data attractor"):
    v1 = []
    v2 = []
    for val in input_data.keys():
        vector = input_data[val]
        v1.append(vector[0])
        v2.append(vector[1])
    plt.plot(v1, v2)
    plt.title(title)
    plt.legend()
    plt.show()
# def plot_attractors():
#     initial_condition = [0, 1, 1]
#     parameters = (t)
#     data_lorenz = odeint(lorenz, initial_condition, parameters)
#     x_data = [x[0] for x in data_lorenz]
#     z_data = [z[2] for z in data_lorenz]
#
#     x_data = normalise_time_series(x_data)
#     z_data = normalise_time_series(z_data)
#
#     plt.plot(x_data, z_data)
#     plt.show()
#
#     initial_condition = [1, 1, 0]
#
#     data_rossler = odeint(rossler, initial_condition, parameters)
#     x_data = [x[0] for x in data_rossler]
#     y_data = [y[1] for y in data_rossler]
#
#     x_data = normalise_time_series(x_data)
#     y_data = normalise_time_series(y_data)
#
#     plt.plot(x_data, y_data)
#     plt.show()
#
#     mackey_glass_data = mackey_glass()
#     x_data, x_delayed_data = couple_pairs(mackey_glass_data)
#     x_data = normalise_time_series(x_data)
#     x_delayed_data = normalise_time_series(x_delayed_data)
#
#     plt.plot(x_data, x_delayed_data)
#     plt.show()
