import numpy as np


# Class which represents a reservoir, the only parameter which it requires is the dimension of the reservoir
class RNN:
    # N represents the dimensionality of the reservoir, i.e the amount of neurons.
    def __init__(self, N):
        self.N = N

        # Reservoir state \in R_N
        self.reservoir = np.random.normal(0, 1, size=(self.N, 1))

        # Connection weights W \in R_{NxN}
        self.connection_weights = self.init_connection_weights()

        # Input weights W_in \in R_{Nx1}
        self.input_weights = np.random.normal(0, 1, size=(self.N, 1))

        # Output weights W_out \in W_{1xL}
        self.output_weights = np.random.normal(0, 1, size=(1, self.N))

        # bias vector, b \in R_N
        self.bias = np.random.normal(0, 1, size=(self.N, 1))

    # Initialise Connection weights, W \in R_{NxN}, w_{ij} reflects connection strength from x_j to x_i [0,1]
    def init_connection_weights(self):
        connection_weights = []

        # The matrix has N rows
        for i in range(self.N):
            row = []
            # The matrix has N columns
            for j in range(self.N):
                w_ij = np.random.normal(0.5, 0.5)
                # weights should be normally distributed [0,1]
                while w_ij < 0 or (w_ij > 1):
                    w_ij = np.random.normal(0.5, 0.5)
                row.append(w_ij)
            connection_weights.append(row)
        # Convert to
        connection_weights = np.matrix(connection_weights)
        return connection_weights

    def get_output(self):
        return np.matmul(self.output_weights, self.reservoir)

    def drive(self):
        return np.matmul(self.connection_weights, self.reservoir)





