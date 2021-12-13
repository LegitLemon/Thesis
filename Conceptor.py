import numpy as np

class Conceptor:
    def __init__(self, R, alpha, N):
        self.N = N
        self.alpha = alpha
        self.R = R
        self.C = None
        self.init_Conceptor_gradient()

    #C(R, α) = R (R + α−2 I)−1 = (R + α−2 I)−1 R
    def init_Conceptor_analytic(self):
        print("Initialisaing Conceptor")
        C_0 = self.alpha**-2 * np.identity(self.N)
        C_1 = np.add(self.R, C_0)
        C_2 = np.linalg.inv(C_1)
        C_3 = np.matmul(self.R, C_2)
        self.C = C_3

    def init_Conceptor_gradient(self):
        self.C = np.random.normal(0, 1, (self.N, self.N))


