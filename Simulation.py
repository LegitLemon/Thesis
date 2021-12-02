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
        self.test = []

    # Samples a simple signal
    def init_pattern(self):
        # Consider the following signal p(n)=sin(2pin/(10sqrt())) sampled for 1000 steps of n
        return [np.sin(2*n/(10*np.sqrt(2))) for n in range(self.T)]

    # updates the state of the RNN
    def next_step(self, p):
        # get autonomous part of new state
        undriven_new = self.rnn.drive()


        # Caution!!!!!!!! Only works for on dimensional input
        # get driven part of new state, driven new seems to work
        driven_new = self.rnn.input_weights*p
        # perform state update equation
        self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias) #self.bias

    def run(self):
        for idx, p in enumerate(self.pattern):
            self.next_step(p)
            self.test.append(float(self.rnn.reservoir[1]))


