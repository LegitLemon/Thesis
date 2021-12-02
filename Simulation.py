from RNN import RNN
import numpy as np

# This class combines a RNN together with a pattern, this class allows (autonomous) driving of the pattern
# with or without an input pattern.
class Simulation:

    # Pattern: discrete time signal of a sample pattern, in this case assumed to be 1 dimensional
    def __init__(self, N, T):
        # Rnn used throughout the simulations
        self.rnn = RNN(N)

        # The amount of timesteps we let the simulation run
        self.T = T
        self.pattern = self.init_pattern()

    # Samples a simple signal
    def init_pattern(self):
        return 1
        pass
    # Performs the update equation, updates the state of the RNN
    def next_step(self):
        # get autonomous part of new state
        undriven_new = self.rnn.drive()

        # Caution!!!!!!!! Only works for on dimensional input
        # get driven part of new state
        driven_new = self.pattern * self.rnn.input_weights

        # perform state update equation
        self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias)

