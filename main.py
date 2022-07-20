from Simulation import Simulation
import matplotlib.pyplot as plt

def main():
    T = 1500
    washout_time = 500
    N_neurons = 100
    aperture = 100

    simulation = Simulation(N=N_neurons, T=T, washout_time=washout_time, aperture=aperture)

    # do you want to store the patterns in the reservoir
    # load patterns into reservoir
    simulation.load(load_patterns=True)

    #tr = simulation.test(with_conceptor=True)
    tr, test1, test2, test3 = simulation.autonomous()
    patterns = simulation.init_patterns()

    for itr, i in enumerate(tr):
        #plt.plot(test1[itr], label="test1: Neuron activation")
        #plt.plot(test2[itr], label="test: same as test 1")
        #plt.plot(test3[itr], label="test: same as test 1")
        plt.plot(i, label="output")
        plt.plot(patterns[itr], label="signal to be retrieved")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
