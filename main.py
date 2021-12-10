from Simulation import Simulation
import matplotlib.pyplot as plt

def main():

    T = 1500
    washout_time = 500
    N_neurons = 100
    aperture = 1000

    simulation = Simulation(N=N_neurons, T=T, washout_time=washout_time, aperture=aperture)

    simulation.load()
    tr = simulation.autonomous()

    t = [x for x in range(1000)]
    patterns = simulation.init_patterns()

    for itr, i in enumerate(tr):
        plt.plot(t, i, label="output")
        plt.plot(patterns[itr], label="target")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
