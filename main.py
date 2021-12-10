from Simulation import Simulation
import matplotlib.pyplot as plt

def main():

    T = 1500
    t = [x for x in range(1000)]

    simulation = Simulation(150, T)
    simulation.load()
    tr = simulation.autonomous()
    patterns = simulation.init_patterns()
    for itr, i in enumerate(tr):
        plt.plot(t, i, label="output")
        plt.plot(patterns[itr], label="target")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
