from brian2 import *
import neuronDynamics as nd

class OutputPopulation():
    def __init__(self):
        tau = 30 * ms

        print("Constructing output population")
        self.outputPopulation = NeuronGroup(N=nd.N_output, model=nd.eqsOut, threshold=nd.thresOut, refractory=nd.refracOut, reset=nd.resetOut)
        self.spikeMonitor = SpikeMonitor(self.outputPopulation)
        print("constructed output population")

    def getReadout(self, t):
        trains = self.spikeMonitor.spike_trains()
        print(trains)
