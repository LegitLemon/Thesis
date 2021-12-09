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

        self.X = None
        self.P = None

        self.state_collection_matrices = []
        self.state_correlation_matrices = []
        self.delayed_state_matrices = []
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
        W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
        return W_opt

    def compute_output_weights(self):
        #append all state collection matrices
        X = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.state_collection_matrices):
            if i != 0:
                X = np.hstack((X, X_j))

        self.X = X
        # append all pattern matrices
        P = np.array(self.patterns[0][-1000:])
        for i, P_j in enumerate(self.patterns):
            if i != 0:
                P = np.hstack((P, np.array(P_j[-1000:])))

        self.P = P
        self.rnn.output_weights = self.ridge_regression(X, P)
        print(self.rnn.output_weights)

    def get_bias_matrix(self):
        b = np.array(self.rnn.bias)
        B = b
        for i in range(3999):
            B = np.hstack((B,b))
        return B
    # W = (( ̃X ̃X′ + rho I_N×N)^−1  ̃X(tanh^−1(X)−B))′
    def store(self):
        X_tilde = self.state_collection_matrices[0]
        rho = 0.0001
        for i, X_j in enumerate(self.state_collection_matrices):
            if i != 0:
                X_tilde = np.hstack((X_tilde, X_j))

        W_1 = np.linalg.inv((np.matmul(X_tilde, X_tilde.transpose()) + rho*np.identity(self.N))).transpose()
        W_2 = np.matmul(W_1, X_tilde)

        B = self.get_bias_matrix()

        ## very sketchy ??
        W_3 = np.subtract(self.X, B).transpose()

        print(W_1.shape, W_2.shape, W_3.shape)
        W_opt = np.matmul(W_2, W_3).transpose()
        self.rnn.connection_weights = W_opt


    def load(self):
        for j, pattern in enumerate(self.patterns):
            for idx, p in enumerate(pattern):
                self.next_step(p)
                self.test.append(float(self.rnn.reservoir[1]))
                state = np.array(self.rnn.reservoir)
                if idx >= 500:
                    if idx == 500:
                        state_matrix = state
                    else:
                        state_matrix = np.c_[state_matrix, state]
                    if idx == 501:
                        delayed_state = state
                    if idx >501:
                        delayed_state = np.c_[delayed_state, state]

            self.state_correlation_matrices.append(np.corrcoef(state_matrix))
            self.delayed_state_matrices.append(delayed_state)
            self.state_collection_matrices.append(state_matrix)

        self.compute_output_weights()
        self.store()

