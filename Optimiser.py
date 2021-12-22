import numpy as np
# class which implements ridge regression used at several times to compute optimal solutions
class Optimiser:
    def __init__(self, rnn, patterns, washout, N, T):

        self.state_collection_matrices = []
        self.delayed_state_matrices = []
        self.rnn = rnn
        self.patterns = patterns
        self.washout_time = washout
        self.N = N
        self.T = T

        self.X = None
        self.P = None

        self.rho_Wout = 0.01
        self.rho_W = 0.01
        self.lamb = 0.001

    # perform a ridge regression with constant rho, and return the analytic solution
    def ridge_regression(self, X, P, rho):
        W_1 = np.linalg.inv((np.dot(X, X.transpose()) + rho*np.identity(self.N)))
        W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
        return W_opt

    def compute_output_weights(self):
        print("Computing optimal output weights")
        #append all state collection matrices
        X = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.state_collection_matrices):
            if i != 0:
                X = np.hstack((X, X_j))
        self.X = X

        # append all pattern matrices
        P = np.array(self.patterns[0][-(self.T-self.washout_time):])
        for i, P_j in enumerate(self.patterns):
            if i != 0:
                P = np.hstack((P, np.array(P_j[-(self.T-self.washout_time):])))

        self.P = P
        return self.ridge_regression(X, P, self.rho_Wout)

    def compute_connection_weights(self):
        print("Computing optimal starting weights")
        X_tilde = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.delayed_state_matrices):
            if i > 0:
                X_tilde = np.hstack((X_tilde, X_j))

        B = self.get_bias_matrix()
        W_opt = self.ridge_regression(X_tilde, B, self.rho_W)
        return W_opt

    def get_bias_matrix(self):
        b = np.array(self.rnn.bias)
        B = b
        for i in range((len(self.patterns)*(self.T-self.washout_time))-1):
            B = np.hstack((B, b))
        return B

    def update_conceptor(self, j):
        alpha = self.rnn.conceptors[0].alpha
        term1 = self.rnn.reservoir - np.dot(self.rnn.conceptors[j].C, self.rnn.reservoir.dot(self.rnn.reservoir.transpose()))
        term2 = alpha**-2*self.rnn.conceptors[j].C
        self.rnn.conceptors[j].C += self.lamb*(term1-term2)
