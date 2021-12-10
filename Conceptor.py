import numpy as np

class Conceptor:

    def __init__(self, R, alpha, N):
        self.N = N
        self.alpha = alpha
        self.R = R
        self.C = None
        self.init_Conceptor()

    #C(R, α) = R (R + α−2 I)−1 = (R + α−2 I)−1 R
    def init_Conceptor(self):
        C_1 = np.linalg.inv(self.R + (self.alpha**-2)*np.identity(self.N))
        self.C = np.matmul(C_1, self.R)


