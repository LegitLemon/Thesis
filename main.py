from Simulation import Simulation
import matplotlib.pyplot as plt

def main():
    T = 1500
    washout_time = 500
    N_neurons = 100
    aperture = 10

    simulation = Simulation(N=N_neurons, T=T, washout_time=washout_time, aperture=aperture)

    # load patterns into reservoir
    simulation.load(loaded=False)

    # compute conceptors
    simulation.load(loaded=True)

    #
    tr = simulation.test()

    patterns = simulation.init_patterns()

    for itr, i in enumerate(tr):
        plt.plot(i, label="output")
        plt.plot(patterns[itr], label="target")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
