# class representing the liquid in the LSM implementation
from brian2 import *
import neuronDynamics as nd
class Liquid():

    def __init__(self):
        self.liquid = NeuronGroup(N=nd.N_liquid, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)
        pass

    def getLiquidState(self):
        pass