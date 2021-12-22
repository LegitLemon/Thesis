import numpy as np

class Conceptor:
    def __init__(self, R, alpha, N):
        self.N = N
        self.alpha = alpha
        self.R = R
        self.C = None
        self.init_Conceptor_analytic()

    #C(R, α) = R (R + α−2 I)−1 = (R + α−2 I)−1 R
    def init_Conceptor_analytic(self):
        print("Initialisaing Conceptor")
        C_0 = self.alpha**-2 * np.identity(self.N)
        self.C = np.dot(self.R, np.linalg.inv(np.add(self.R, C_0)))

    def init_Conceptor_gradient(self):
        self.C = np.random.normal(1, 1, (self.N, self.N))


