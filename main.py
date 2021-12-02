import numpy as np

from Simulation import Simulation

def main():



    T = 100
    simulation = Simulation(10, T)
    simulation.run()
    print(simulation.rnn.reservoir)



if __name__ == "__main__":
    main()
