import numpy as np
import LSM.neuronDynamics as nd
from Classification.Conceptor import Conceptor
class Classifier:
    def __init__(self):
        self.alpha = 0.5
        self.stateMatrices = []
        self.inputSpikeTrains = []
        self.conceptors = []
        self.correlations = []

    def computeConceptors(self):
        print("Computing Conceptors")
        for stateMatrix in self.stateMatrices:
            R = np.dot(stateMatrix, stateMatrix.transpose()) / nd.amountOfRunsPerPattern
            print(R.shape)
            self.conceptors.append(Conceptor(R, self.alpha, nd.N_liquid))

    def computePositiveEvidence(self, z):
        evidences = []
        for C in self.conceptors:
            evidences.append(np.dot(np.dot(z.transpose(), C), z))
        return evidences

    def classify(self, z):
        evidences = self.computePositiveEvidence(z)
        judgement = np.argmax(np.array(evidences))
        return judgement
