from brian2 import *
import neuronDynamics as nd

class OutputPopulation():
    def __init__(self):
        self.outputPopulation = NeuronGroup(N=nd.N_output, model=nd.eqsOut, refractory=nd.refracOut, reset=nd.resetOut)

    def getReadout(self):
        pass
