from brian2 import *
import LSM.neuronDynamics as nd
class OutputPopulation:
    def __init__(self):
        print("Constructing output population")
        self.outputPopulation = NeuronGroup(N=nd.N_output, model=nd.eqsOut, threshold=nd.thresOut, refractory=nd.refracOut)
        self.spikeMonitor = SpikeMonitor(self.outputPopulation)
        print("constructed output population")

    def computeBinnedActivity(self):
        trains = self.spikeMonitor.spike_trains()
        binnedActivity = []
        for neuron in trains.keys():
            print(neuron)
            currentSpiketrain = trains[neuron]
            binnedActivity.append(self.computeBinnedSpiketrain(currentSpiketrain))

    def computeBinnedSpiketrain(self, spiketrain):
        binnedActivity = []
        upperbound = 0 * ms
        lowerbound = nd.binSize
        while(upperbound < nd.simLength):
            spikeCount = 0
            for spike in spiketrain:
                if spike >= lowerbound and spike <= upperbound:
                    spikeCount += 1

            spikerate = spikeCount / (nd.binSize/ms)
            binnedActivity.append(spikerate)
            upperbound += nd.binSize
            lowerbound += nd.binSize
        return binnedActivity

