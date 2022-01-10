import random

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
    def next_step_with_input(self, p=0, C=None):
        # get autonomous part of new state
        undriven_new = self.rnn.drive_with_input()
        # get driven part of new state, driven new seems to work
        driven_new = self.rnn.input_weights*p
        # perform state update equation
        if C is not None:
            self.rnn.reservoir = np.dot(C, np.tanh(undriven_new + driven_new + self.rnn.bias))
        else:
            self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias)


    def next_step_without_input(self, C):
        undriven_new = self.rnn.drive_with_input()
        self.rnn.reservoir = np.dot(C, np.tanh(undriven_new+self.rnn.bias))

    def load(self, load_patterns=True):
        print("Began storing patterns")
        state_matrix = None
        delayed_state = None
        for j, pattern in enumerate(self.patterns):
            self.rnn.init_reservoir()
            print("Driving pattern: ", j)
            for idx, p in enumerate(pattern):
                state = np.array(self.rnn.reservoir)
                if idx == self.washout_time:
                    state_matrix = state
                if idx > self.washout_time:
                    state_matrix = np.c_[state_matrix, state]
                if idx == self.washout_time-1:
                    delayed_state = np.array(self.rnn.reservoir)
                if idx > self.washout_time-1 and idx< self.T-1:
                    delayed_state = np.c_[delayed_state, state]
                self.next_step_with_input(p)
            print("finished driving pattern: ", j)
            print(state.shape)
            R = np.dot(state_matrix, state_matrix.transpose()) / self.N
            self.rnn.conceptors.append(Conceptor(R, self.alpha, self.N))
            self.optimizer.state_collection_matrices.append(state_matrix)
            self.optimizer.delayed_state_matrices.append(delayed_state)

        self.rnn.output_weights = self.optimizer.compute_output_weights()
        if load_patterns:
            self.rnn.connection_weights = self.optimizer.compute_connection_weights()

    # retrieval using conceptors
    def autonomous(self):
        test1, test2, test3 = [], [] ,[]
        n1, n2, n3 = random.randrange(0, 99), random.randrange(0, 99), random.randrange(0, 99)
        ts = []
        for j in range(len(self.patterns)):
            self.rnn.init_reservoir()
            test_run = []
            ts1, ts2, ts3 = [], [], []
            for n in range(self.T):
                self.next_step_without_input(self.rnn.conceptors[j].C)
                test_run.append(self.rnn.get_output())
                ts1.append(self.rnn.reservoir[n1])
                ts2.append(self.rnn.reservoir[n2])
                ts3.append(self.rnn.reservoir[n3])
            ts.append(test_run)
            test1.append(ts1)
            test2.append(ts2)
            test3.append(ts3)
        return ts, test1, test2, test3

    # check whether the pattern has been encoded in the reservoir some washout time
    def test(self, with_conceptor=False):
        ts = []
        for j, pattern in enumerate(self.patterns):
            test_run = []
            for n, p_n in enumerate(pattern):
                if with_conceptor:
                    self.next_step_with_input(p_n, self.rnn.conceptors[j].C)
                else:
                    self.next_step_with_input(p_n)
                test_run.append(self.rnn.get_output())
            ts.append(test_run)
        return ts



