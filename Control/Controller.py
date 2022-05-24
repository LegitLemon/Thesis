from Simulation import Simulation
import LSM.neuronDynamics as nd
from Control.Optimiser import Optimiser
from brian2 import *
from Utils.ProgressBar import ProgressBar
from Classification.Conceptor import Conceptor

class Controller:
    def __init__(self):
        self.simulation = Simulation(True)
        patterns = self.simulation.signalEncoder.spikedPatterns[0]
        binnedPattern = self.simulation.liquid.computeBinnedSpiketrain(patterns * ms)
        print(binnedPattern)
        self.optimiser = Optimiser([np.array(binnedPattern)])


        self.outputWeights = None
        self.loadingWeights = None
        self.systemStates = None
        self.conceptor = None
        self.spikeCountsRetrieval = [0]*nd.N_liquid


    def initController(self):
        print("Starting Controller Procedure, initialising conceptor")
        self.simulation.liquid.liquid.v_th[:] = 15*mV
        self.simulation.network.store()
        self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
        self.systemStates = self.simulation.liquid.computeStateMatrixControl()

        stateMatrix, delayedStateMatrix = self.systemMatrixToStateMatrix(np.array(self.systemStates))
        print("Printing StateMatrix")

        self.optimiser.state_collection_matrices.append(np.array(stateMatrix))
        self.optimiser.delayed_state_matrices.append(np.array(delayedStateMatrix))
        self.outputWeights = self.optimiser.compute_output_weights()
        # print(self.outputWeights)
        corMatrix = np.dot(stateMatrix, stateMatrix.transpose()) / nd.N_liquid
        self.loadingWeights = self.optimiser.compute_connection_weights()
        self.conceptor = Conceptor(corMatrix, 0.5, nd.N_liquid)
        # print(self.conceptor.C)
        print("computing optimal connection weights, adjusting network parameters")
        self.simulation.network.restore()
        self.adjustOutputWeights()
        self.adjustInternalWeights()
        self.simulation.network.store()
        self.simulation.run()
        self.conceptorControl()
        self.simulation.signalEncoder.plotEncodedSpikeSignal(0)
        self.simulation.plotRun(0)

    def adjustOutputWeights(self):
        for index, value in enumerate(self.outputWeights):
            self.simulation.outputSynapses.w[index, 0] = value * mvolt
        self.simulation.network.store()

    def adjustInternalWeights(self):
        # print(self.loadingWeights)
        for row, rowVector in enumerate(self.loadingWeights):
            for column, value in enumerate(rowVector):
                # print(value)
                self.simulation.liquid.synapses.w[row, column] = value * mvolt

    def systemMatrixToStateMatrix(self, systemMatrix):
        stateMatrix = systemMatrix[:, nd.washoutTime:]
        delayedStateMatrix = systemMatrix[:, nd.washoutTime-1:-1]
        print(stateMatrix.shape)
        print(delayedStateMatrix.shape)
        return stateMatrix, delayedStateMatrix


    def conceptorControl(self):
        # self.simulation.network.restore()
        # self.simulation.resetInput(0)
        self.simulation.network.restore()
        self.simulation.network.remove(self.simulation.signalEncoder.spikeGenerator)
        self.simulation.network.remove(self.simulation.inputSynapses)
        self.simulation.network.remove(self.simulation.inputMonitor)
        thresholds = []
        for i in range(int((2500/(nd.binSize/ms)-1))):
            print(i*nd.binSize)
            self.simulation.liquid.resetControl()
            self.simulation.network.run(nd.binSize)
            newThresholds = self.updateNeuronThresholds(i)
            thresholds.append(newThresholds)
        self.plotThresholds(thresholds)

    def getCurrentNeuronStates(self, currentIndex):
        liquidSpiketrains = self.simulation.liquid.spikemonitor.spike_trains()
        cutLiquidSpikeTrains = []
        for index, (liquidSpikeTrain, currentCorrespondingSpikecount) in enumerate(zip(liquidSpiketrains.keys(), self.spikeCountsRetrieval)):
            spiketrain = liquidSpiketrains[liquidSpikeTrain]
            liquidSpiketrains[liquidSpikeTrain] = spiketrain[currentCorrespondingSpikecount:]
            self.spikeCountsRetrieval[index] = len(liquidSpiketrains[liquidSpikeTrain]) + self.spikeCountsRetrieval[index]

        binnedLiquidStates = self.simulation.liquid.computeBinnedActivity(liquidSpiketrains, currentIndex*nd.binSize+nd.binSize)
        states = []
        for binnedSpikeTrain in binnedLiquidStates:
            states.append(binnedSpikeTrain[currentIndex])
        return states

    def updateNeuronThresholds(self, binIndex):
        neuronStates = self.getCurrentNeuronStates(binIndex)
        thresholds = []
        for index, currentNeuronState in enumerate(neuronStates):
            row = self.conceptor.C[index, :]
            column = self.conceptor.C[:, index]
            # print(column)
            # print(np.array(neuronStates))
            computedThreshold = np.dot(row, np.array(neuronStates))*mvolt
            if computedThreshold > nd.upperboundThreshold:
                threshold = nd.upperboundThreshold
            elif computedThreshold < nd.lowerboundThreshold:
                threshold = nd.lowerboundThreshold
            else:
                threshold = computedThreshold
            thresholds.append(threshold)
            self.simulation.liquid.liquid.v_th[index] = threshold
        print("membrane potential: ", self.simulation.liquid.liquid.v[3])
        print("firing rate:", neuronStates[3])
        print("threshold: ", thresholds[3])
        print("computed threshold: ", computedThreshold)
        print()
        return thresholds

    def plotThresholds(self, threholds):
        matrix = np.array(threholds)
        matrix = matrix.T
        for i in range(nd.N_liquid):
            thresholds = matrix[i]
            plt.plot(thresholds)
        plt.show()