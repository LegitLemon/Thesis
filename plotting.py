from networkParameters import *
import random
import matplotlib.pyplot as plt
def plot_control_errors(amount):
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
    plt.title("Difference between conceptor projection and current state")
    plt.show()


def plot_liquid_states(trajectories):
    for i in range(6):
        index = random.randint(0, N-1)
        plt.plot(t, trajectories[:, index], label=i)
    plt.plot(t, np.sin(t/20), label="input signal")
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

def plot_output_retrieval(trajectories_projection, trajectories_negation, used_weights):
    output1 = []
    output2 = []
    for state_number in range(len(trajectories_negation)):
        output_point_projection = np.dot(used_weights, trajectories_projection[state_number])
        output1.append(output_point_projection)

        output_point_negation = np.dot(used_weights, trajectories_negation[state_number])
        output2.append(output_point_negation)

    # plt.plot(t, output1)
    plt.plot(t, output1, label="projection retrieval")
    plt.plot(t, output2, label="negation retrieval")
    plt.plot(t, 2*np.sin(t/20), label="goal signal")
    plt.xlabel('time')
    plt.title(f'System output, tau={tau}, a=${a_internal}, N=${N}')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()


