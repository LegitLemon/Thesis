import numpy as np


# Class which represents a reservoir, the only parameter which it requires is the dimension of the reservoir
class RNN:
    # N represents the dimensionality of the reservoir, i.e the amount of neurons.
    def __init__(self, N):
        self.N = N

        # Initialise the three matrices
        self.connection_weights = self.init_reservoir()
        self.input_weights = self.init_input()
        self.output_weights = self.init_output()

    # Initialise
    def init_reservoir(self):



    def init_input(self):
        pass

    def init_output(self):
        pass