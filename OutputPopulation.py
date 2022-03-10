from brian2 import *
import neuronDynamics as nd
from brian2 import *
class OutputPopulation():
    def __init__(self):
        print("Constructing output population")
        self.outputPopulation = NeuronGroup(N=nd.N_output, model=nd.eqsOut, threshold=nd.thresOut, refractory=nd.refracOut)
        self.spikeMonitor = SpikeMonitor(self.outputPopulation)
        print("constructed output population")

    # get the state matrix of a single pattern
    def getStateMatrix(self):
        trains = self.spikeMonitor.spike_trains()
        state_matrix = []
        for idx, spiketrain in enumerate(trains):
            currslot = 20 * ms
            spike_rate_binned = []
            spike_count = 0
            for spike in spiketrain:
                if spike > currslot:
                    spike_rate_during_bin = spike_count / (currslot/ms)
                    spike_rate_binned.append(spike_rate_during_bin)
                    currslot += 20*ms
                spike_count += 1
            state_matrix.append(spike_rate_binned)
        return state_matrix