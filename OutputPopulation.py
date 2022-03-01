from brian2 import *
import neuronDynamics as nd

class OutputPopulation():
    def __init__(self):
        self.outputPopulation = NeuronGroup(N=nd.N_output, model=nd.eqsOut, refractory=nd.refracOut, reset=nd.resetOut)
        self.spikeMonitor = SpikeMonitor(self.outputPopulation)

    def getReadout(self, t):
        trains = self.spikeMonitor.spike_trains()
        print(trains)
