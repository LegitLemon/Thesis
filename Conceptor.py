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
        print("Initialisaing Conceptor")
        C_0 = np.add(self.R, np.add(np.identity(self.N), (self.alpha**-2)))
        C_1 = np.linalg.inv(C_0)
        self.C = C_1.dot(self.R)


