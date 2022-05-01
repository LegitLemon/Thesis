import numpy as np
import LSM.neuronDynamics as nd
# class which implements ridge regression used at several times to compute optimal solutions
class Optimiser:
    def __init__(self, patterns):

        self.state_collection_matrices = []
        self.delayed_state_matrices = []
        self.patterns = patterns
        self.washout_time = nd.washoutTime
        self.N = nd.N_liquid
        self.T = nd.amountOfBins

        self.X = None
        self.P = None

        self.rho_Wout = 0.01
        self.rho_W = 0.1
        self.lamb = 0.001

    # perform a ridge regression with constant rho, and return the analytic solution
    def ridge_regression(self, X, P, rho):
        W_1 = np.linalg.inv((np.dot(X, X.transpose()) + rho*np.identity(self.N)))
        W_opt = (W_1.dot(X).dot(P.transpose())).transpose()
        return W_opt

    def compute_output_weights(self):
        print("Computing optimal output weights")
        X = self.state_collection_matrices[0]
        for i, X_j in enumerate(self.state_collection_matrices):
            if i != 0:
                X = np.hstack((X, X_j))
        self.X = X

        # append all pattern matrices
        P = np.array(self.patterns[0][-(self.T-self.washout_time-1):])
        print(P.shape)
        for i, P_j in enumerate(self.patterns):
            if i != 0:
                P = np.hstack((P, np.array(P_j[-(self.T-self.washout_time):])))
        self.P = P
        return self.ridge_regression(X, P, self.rho_Wout)

    def compute_connection_weights(self):
        print("Computing optimal starting weights")
        X_tilde = self.delayed_state_matrices[0]
        for i, X_j in enumerate(self.delayed_state_matrices):
            if i != 0:
                X_tilde = np.hstack((X_tilde, X_j))
        # B = self.get_bias_matrix()
        val2 = self.X
        # val2 = np.arctanh(self.X)-B
        # print("X: ",self.X.shape)
        # print("B: ", B.shape)
        print("X,tilde: ", X_tilde.shape)
        W_opt = self.ridge_regression(X_tilde, val2, self.rho_W)
        print("Opt: ", W_opt.shape)
        return W_opt

    # def get_bias_matrix(self):
    #     b = np.array(self.rnn.bias)
    #     B = b
    #     for i in range((len(self.patterns)*(self.T-self.washout_time))-1):
    #         B = np.hstack((B, b))
    #     return B
    #
    # def update_conceptor(self, j):
    #     alpha = self.rnn.conceptors[0].alpha
    #     term1 = self.rnn.reservoir - np.dot(self.rnn.conceptors[j].C, self.rnn.reservoir.dot(self.rnn.reservoir.transpose()))
    #     term2 = alpha**-2*self.rnn.conceptors[j].C
    #     self.rnn.conceptors[j].C += self.lamb*(term1-term2)