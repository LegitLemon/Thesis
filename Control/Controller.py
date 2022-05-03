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



    def initController(self):
        print("Starting Controller Procedure, initialising conceptor")
        self.simulation.liquid.liquid.v_th[:] = 15*mV
        self.simulation.network.store()
        self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
        self.systemStates = self.simulation.liquid.computeStateMatrixControl()

        stateMatrix, delayedStateMatrix = self.systemMatrixToStateMatrix(np.array(self.systemStates))

        self.optimiser.state_collection_matrices.append(np.array(stateMatrix))
        self.optimiser.delayed_state_matrices.append(np.array(delayedStateMatrix))
        self.outputWeights = self.optimiser.compute_output_weights()
        print(self.outputWeights)
        self.loadingWeights = self.optimiser.compute_connection_weights()
        self.conceptor = Conceptor(np.corrcoef(stateMatrix), 0.5, nd.N_liquid)
        print("computing optimal connection weights, adjusting network parameters")
        self.simulation.network.restore()
        self.adjustOutputWeights()
        self.adjustInternalWeights()
        self.testTrainingOutputWeights()
        self.conceptorControl()


    def adjustOutputWeights(self):
        for index, value in enumerate(self.outputWeights):
            self.simulation.outputSynapses.w[index, 0] = value * mvolt
        self.simulation.network.store()

    def adjustInternalWeights(self):
        print(self.loadingWeights)
        for row, rowVector in enumerate(self.loadingWeights):
            for column, value in enumerate(rowVector):
                self.simulation.liquid.synapses.w[row, column] = value * mvolt

    def systemMatrixToStateMatrix(self, systemMatrix):
        stateMatrix = systemMatrix[:, nd.washoutTime:]
        delayedStateMatrix = systemMatrix[:, nd.washoutTime-1:-1]
        print(stateMatrix.shape)
        print(delayedStateMatrix.shape)
        return stateMatrix, delayedStateMatrix

    def testTrainingOutputWeights(self):
        self.simulation.run()

    def initNeuronThresholdNormal(self):
        self.simulation.liquid.liquid.v_th[:] = 15*mV


    def conceptorControl(self):
        pass
