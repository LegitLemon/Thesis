from RNN import RNN
import numpy as np
from scipy import signal

# This class combines a RNN together with a pattern, this class allows (autonomous) driving of the pattern
# with or without an input pattern.
class Simulation:

    # Pattern: discrete time signal of a sample pattern, in this case assumed to be 1 dimensional
    def __init__(self, N, T):
        # Rnn used throughout the simulations
        self.N = N
        self.rnn = RNN(N)

        # The amount of timesteps we let the simulation run
        self.T = T
        self.patterns = self.init_patterns()

        self.state_collection_matrices = []
        self.state_correlation_matrices = []
        self.test = []

    # Samples a simple signal
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
        undriven_new = self.rnn.drive()

        # get driven part of new state, driven new seems to work
        driven_new = self.rnn.input_weights*p
        # perform state update equation
        self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias) #self.bias

    def ridge_regression(self, X, P):
        rho = 0.01

        # W out = ((XX^T + (roh_out)I_N×N )^−1 X P^T)^T
        print(X.shape)
        W_1 = np.linalg.inv((np.matmul(X, X.transpose()) + rho*np.identity(self.N)))
        W_opt =(W_1.dot(X).dot(P.transpose())).transpose()
        return W_opt

    def compute_output_weights(self):
        #append all state collection matrices
        X = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.state_collection_matrices):
            if i != 0:
                X = np.hstack((X, X_j))

        # append all pattern matrices
        P = np.array(self.patterns[0][-1000:])
        for i, P_j in enumerate(self.patterns):
            if i != 0:
                P = np.hstack((P, np.array(P_j[-1000:])))

        self.rnn.output_weights = self.ridge_regression(X, P)

    def load(self):
        for j, pattern in enumerate(self.patterns):
            for idx, p in enumerate(pattern):
                self.next_step(p)
                self.test.append(float(self.rnn.reservoir[1]))
                state = np.array(self.rnn.reservoir)
                if idx > 499:
                    if idx == 500:
                        state_matrix = state
                    else:
                        state_matrix = np.c_[state_matrix, state]
            self.state_correlation_matrices.append(np.corrcoef(state_matrix))
            self.state_collection_matrices.append(state_matrix)


        self.compute_output_weights()

