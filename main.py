import numpy as np

from Simulation import Simulation

def main():

    # Consider the following signal p(n)=sin(2pin/(10sqrt()))


    simulation = Simulation(10, 2)
    simulation.next_step()
    print(simulation.rnn.reservoir)



if __name__ == "__main__":
    main()
