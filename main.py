from Simulation import Simulation
import matplotlib.pyplot as plt

def main():



    T = 2000
    t = [x for x in range(T)]

    simulation = Simulation(100, T)
    simulation.run()

    plt.plot(t, simulation.pattern, label="2")
    plt.plot(t, simulation.test, label="1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
