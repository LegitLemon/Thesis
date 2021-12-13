from RNN import RNN
import numpy as np
from scipy import signal
from Conceptor import Conceptor
from Optimiser import Optimiser


# This class combines a RNN together with a pattern, this class allows (autonomous) driving of the pattern
class Simulation:
    # Pattern: discrete time signal of a sample pattern, in this case assumed to be 1 dimensional
    def __init__(self, N, T, washout_time, aperture):
        self.N = N
        self.rnn = RNN(N)
        self.alpha = aperture
        self.washout_time = washout_time

        # The amount of timesteps we let the simulation run
        self.T = T
        self.patterns = self.init_patterns()
        self.optimizer = Optimiser(self.rnn, self.patterns, self.washout_time, self.N, self.T)

    # Sample 4 different one dimensional signals
    def init_patterns(self):
        patterns = []
        # Consider the following signal p(n)=sin(2pin/(10sqrt())) sampled for 1000 steps of n
        patterns.append([np.sin(2*n/(10*np.sqrt(2))) for n in range(self.T)])

        # Square Wave
        patterns.append([signal.square(n/10) for n in range(self.T)])

        # regular cosine
        patterns.append([np.cos(n/10) for n in range(self.T)])

        # Sawtooth Wave
        patterns.append([signal.sawtooth(n/10) for n in range(self.T)])

        return patterns

    # updates the state of the RNN
    def next_step(self, p):
        # get autonomous part of new state
        undriven_new = self.rnn.drive_with_input()
        # get driven part of new state, driven new seems to work
        driven_new = self.rnn.input_weights*p
        # perform state update equation
        self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias)

    def load(self, loaded):
        print("Began storing patterns")
        for j, pattern in enumerate(self.patterns):
            self.rnn.init_reservoir()
            print("Driving pattern: ", j)
            for idx, p in enumerate(pattern):
                self.next_step(p)
                state = np.array(self.rnn.reservoir)
                if idx >= self.washout_time:
                    if idx == self.washout_time:
                        state_matrix = state
                    else:
                        state_matrix = np.c_[state_matrix, state]
                    if idx == self.washout_time+1:
                        delayed_state = state
                    if idx >self.washout_time+1:
                        delayed_state = np.c_[delayed_state, state]

            print("finished driving pattern: ", j)
            delayed_state = np.c_[delayed_state, state]

            if loaded is True:
                R = np.corrcoef(state_matrix)
                self.rnn.conceptors.append(Conceptor(R, self.alpha, self.N))
            else:
                self.optimizer.state_collection_matrices.append(state_matrix)
                self.optimizer.delayed_state_matrices.append(delayed_state)

        if loaded is False:
            self.rnn.output_weights = self.optimizer.compute_output_weights()
            self.rnn.connection_weights = self.optimizer.compute_connection_weights()

    # retrieval using conceptors
    def autonomous(self):
        ts = []
        for j in range(len(self.patterns)):
            self.rnn.init_reservoir()
            test_run = []
            for n in range(1000):
                self.rnn.reservoir = np.matmul(self.rnn.conceptors[j].C, np.tanh(self.rnn.drive_with_input())+self.rnn.bias)
                test_run.append(self.rnn.get_output())
            ts.append(test_run)
        return ts

    # check whether the pattern has been encoded in the reservoir some washout time
    def test(self):
        ts = []
        for j, pattern in enumerate(self.patterns):
            test_run = []
            for p_n, n in enumerate(pattern):
                self.next_step(n)
                test_run.append(self.rnn.get_output())
            ts.append(test_run)
        return ts



