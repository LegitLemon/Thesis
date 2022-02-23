from brian2 import *
import neuronDynamics as nd

class OutputPopulation():
    def __init__(self):
        self.outputPopulation = NeuronGroup(N=nd.N_liquid, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)

    def getReadout(self):
        pass
