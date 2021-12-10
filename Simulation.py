from RNN import RNN
import numpy as np
from scipy import signal
from Conceptor import Conceptor

# This class combines a RNN together with a pattern, this class allows (autonomous) driving of the pattern
# with or without an input pattern.
class Simulation:

    # Pattern: discrete time signal of a sample pattern, in this case assumed to be 1 dimensional
    def __init__(self, N, T):
        # Rnn used throughout the simulations
        self.N = N
        self.rnn = RNN(N)
        self.alpha = 100
        self.washout_time = 500
        # The amount of timesteps we let the simulation run
        self.T = T
        self.patterns = self.init_patterns()

        self.X = None
        self.P = None

        self.state_collection_matrices = []
        self.delayed_state_matrices = []

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
        undriven_new = self.rnn.drive_with_input()

        # get driven part of new state, driven new seems to work
        driven_new = self.rnn.input_weights*p
        # perform state update equation
        self.rnn.reservoir = np.tanh(undriven_new + driven_new + self.rnn.bias)

    def ridge_regression(self, X, P):
        rho = 0.01
        # W out = ((XX^T + (roh_out)I_N×N )^−1 X P^T)^T
        print(X.shape)
        W_1 = np.linalg.inv((np.matmul(X, X.transpose()) + rho*np.identity(self.N)))
        W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
        return W_opt

    def compute_output_weights(self):
        print("Computing optimal output weights")
        #append all state collection matrices
        X = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.state_collection_matrices):
            print(X_j.shape)
            if i != 0:
                X = np.hstack((X, X_j))
        print(X.shape)
        self.X = X
        # append all pattern matrices
        P = np.array(self.patterns[0][-1000:])
        for i, P_j in enumerate(self.patterns):
            if i != 0:
                P = np.hstack((P, np.array(P_j[-1000:])))

        self.P = P
        self.rnn.output_weights = self.ridge_regression(X, P)

    def get_bias_matrix(self):
        b = np.array(self.rnn.bias)
        B = b
        for i in range((len(self.patterns)*1000)-1):
            B = np.hstack((B,b))
        return B

    # W = (( ̃X ̃X′ + rho I_N×N)^−1  ̃X(tanh^−1(X)−B))′
    def store(self):
        print("Computing optimal starting weights")
        X_tilde = self.state_collection_matrices[0]
        rho = 0.0001
        for i, X_j in enumerate(self.delayed_state_matrices):
            if i > 0:
                X_tilde = np.hstack((X_tilde, X_j))

        W_1 = np.linalg.inv((np.matmul(X_tilde, X_tilde.transpose()) + rho*np.identity(self.N)))
        #W_2 = np.matmul(W_1, X_tilde)

        B = self.get_bias_matrix()

        ## very sketchy ??
        W_2 = np.subtract(np.arctan(self.X), B).transpose()
        W_3 = np.matmul(X_tilde, W_2)
        print(W_1.shape, W_2.shape, W_3.shape)
        W_opt = np.matmul(W_1, W_3).transpose()
        self.rnn.connection_weights = W_opt

    def test(self):
        ts =[]
        for j, pattern in enumerate(self.patterns):
            test_run =[]
            for p_n, n in enumerate(pattern):
                self.next_step(n)
                test_run.append(self.rnn.get_output())
            ts.append(test_run)
        return ts

    def load(self):
        print("Began storing patterns")
        for j, pattern in enumerate(self.patterns):
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
            R = np.corrcoef(state_matrix)
            self.rnn.conceptors.append(Conceptor(R, self.alpha, self.N))
            self.delayed_state_matrices.append(delayed_state)
            self.state_collection_matrices.append(state_matrix)

        self.compute_output_weights()
        self.store()

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


