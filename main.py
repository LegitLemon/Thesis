from Simulation import Simulation
import matplotlib.pyplot as plt

def main():

    T = 1500
    t = [x for x in range(T)]

    simulation = Simulation(100, T)
    simulation.load()
    for i in range(len(simulation.patterns)):
        plt.plot(t, simulation.patterns[i])
        #plt.plot(t, simulation.test, label="reservoir activation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
