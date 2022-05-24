# class representing the liquid in the LSM implementation
from brian2 import *
import LSM.neuronDynamics as nd
import random

class Liquid():

    def __init__(self, control):
        print("Making new LSM")
        self.neurontypes = self.initNeuronTypes()

        if control is True:
            self.liquid = NeuronGroup(N=nd.N_liquid, threshold=nd.thresOutDynamicThreshold,
                                      model=nd.eqsDynamicThreshold, refractory=nd.refrac, reset=nd.resetDynamicThreshold)
            self.initTimeConstants()
            self.thresholdMonitor = StateMonitor(self.liquid, 'v_th', record=np.random.randint(0, nd.N_liquid, 5))

        else:
            self.liquid = NeuronGroup(N=nd.N_liquid, threshold=nd.thres, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)

        self.synapses = Synapses(self.liquid, self.liquid, model="w:volt", on_pre=nd.weightEQ)
        self.spikemonitor = SpikeMonitor(self.liquid)
        self.stateMonitor = StateMonitor(self.liquid, 'v', record=np.random.randint(0, nd.N_liquid, 5))

        self.connectionCount = 0
        self.count = 0
        print("starting LSM synapses")
        self.synapses.connect(p=nd.connecProb)
        self.synapses.w[:, :] = '0.5*rand()*mvolt'

        amount = int(nd.proportionInhib*nd.N_liquid)
        indeces = []
        for x in range(amount):
            ind = random.randint(0, nd.N_liquid-1)

            while ind in indeces:
                ind = random.randint(0, nd.N_liquid-1)
            indeces.append(ind)
            self.synapses.w[ind, :] = '-0.02*rand()*mvolt'


    def initTimeConstants(self):
        for i in range(nd.N_liquid):
            self.liquid.tau[i] = random.randint(25, 30) *ms

    def initNeuronTypes(self):
        # initialise all neurons to be excitatory
        neuronTypes = [False for x in range(nd.N_liquid)]
        # compute the amount of inhibitory connections
        amount = int(0.2*nd.N_liquid)
        # Set that amount of neurons to inhibitory
        indeces = []
        for x in range(amount):
            ind = random.randint(0, nd.N_liquid-1)

            while ind in indeces:
                ind = random.randint(0, nd.N_liquid-1)
            indeces.append(ind)
            neuronTypes[ind] = True
        return neuronTypes

    def getDistance(self, a, b):
        diff = 0
        for i in range(len(a)):
            diff += (a[i]-b[i])**2
        return sqrt(diff)

    def initSynapses(self):
        number = 0
        for z in range(9):
            print(number)
            for y in range(3):
                for x in range(3):
                    number += 1
                    self.initNeuron((x, y, z), number)

    def initNeuron(self, p1, neuronFrom):
        neuronTo = 0
        for z in range(9):
            for y in range(3):
                for x in range(3):
                    neuronTo += 1
                    dist = self.getDistance(p1, (x, y, z))
                    self.setConncetion(dist, neuronFrom, neuronTo)

    def setConncetion(self, dist, neuronFrom, neuronTo):
        prob = 0.5
        val = random.random()
        self.count += 1
        if val < prob:
            self.connectionCount += 1
            value = np.random.normal(loc=nd.liquidSynapseStrength, scale=0.5*nd.liquidSynapseStrength)
            if self.neurontypes[neuronFrom] == True:
                value *= -1
            self.synapses.connect(i=neuronFrom, j=neuronTo)
            self.synapses.w[neuronFrom, neuronTo] = value*mV

    def reset(self):
        for i in range(nd.N_liquid):
            self.liquid.v[i] = random.uniform(13.5, 15) * mV

    def computeStateMatrixClassification(self, inputSpikeTrain):
        liquidSpiketrains = self.spikemonitor.spike_trains()
        binnedLiquidStates = self.computeBinnedActivity(liquidSpiketrains)
        binnedInputStates = self.computeBinnedActivity({"1": inputSpikeTrain})[0]
        return self.compileStateMatrixClassification(binnedInputStates, binnedLiquidStates)

    def compileStateMatrixClassification(self, inputTrain, liquidTrain):
        stateVector = []
        for binIndex in range(len(inputTrain)):
            for i in range(len(liquidTrain)):
                stateVector.append(liquidTrain[i][binIndex])
            stateVector.append(inputTrain[binIndex])
        return stateVector

    def resetControl(self):
        for i in range(nd.N_liquid):
            self.liquid.v[i] = random.uniform(8, 16.5) * mV


    def computeStateMatrixControl(self):
        liquidSpiketrains = self.spikemonitor.spike_trains()
        return self.computeBinnedActivity(liquidSpiketrains)

    def computeBinnedActivity(self, trains, finalTimeStep=None):
        binnedActivity = []
        for neuron in trains.keys():
            currentSpiketrain = trains[neuron]
            binnedActivity.append(self.computeBinnedSpiketrain(currentSpiketrain, finalTimeStep))
        return binnedActivity

    def computeBinnedSpiketrain(self, spiketrain, finalTimeStep=None):
        binnedActivity = []
        upperbound = nd.binSize
        lowerbound = 0*ms

        if finalTimeStep is not None:
            finalUpperBound = finalTimeStep
        else:
            finalUpperBound = nd.simLength

        while(upperbound <= nd.simLength):
            # if lowerbound == nd.simLength:
            #     break
            spikeCount = 0
            for spike in spiketrain:
                if spike >= lowerbound and spike <= upperbound:
                    spikeCount += 1
            spikerate = spikeCount / (nd.binSize)
            binnedActivity.append(spikerate)
            upperbound += nd.binSize
            lowerbound += nd.binSize
        return binnedActivity


