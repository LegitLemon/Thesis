from Simulation import Simulation
import LSM.neuronDynamics as nd
from Control.Optimiser import Optimiser
from brian2 import *
from Utils.ProgressBar import ProgressBar
from Classification.Conceptor import Conceptor

class Controller:
    def __init__(self):
        self.simulation = Simulation()
        patterns = self.simulation.signalEncoder.spikedPatterns[0]
        binnedPattern = self.simulation.liquid.computeBinnedSpiketrain(patterns * ms)
        print(binnedPattern)
        self.optimiser = Optimiser([np.array(binnedPattern)])

        self.outputWeights = None
        self.loadingWeights = None
        self.stateMatrix = None
        self.conceptor = None

    def initController(self):
        print("Starting Controller Procedure, initialising conceptor")
        self.simulation.network.store()
        self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
        self.stateMatrix = self.simulation.liquid.computeStateMatrixControl()
        self.optimiser.state_collection_matrices.append(np.array(self.stateMatrix))
        self.outputWeights = self.optimiser.compute_output_weights()
        print(self.outputWeights)
        # self.loadingWeights = self.optimiser.compute_connection_weights()
        # self.conceptor = Conceptor(np.corrcoef(self.stateMatrix))
        print("computing optimal connection weights, adjusting network parameters")
        self.simulation.network.restore()
        self.adjustOutputWeights()
        self.testTrainingOutputWeights()

    def adjustOutputWeights(self):
        for index, value in enumerate(self.outputWeights):
            self.simulation.outputSynapses.w[index, 0] = value * mvolt
        self.simulation.network.store()

    def testTrainingOutputWeights(self):
        self.simulation.run()

    def inputToRateBasedState(self, inputSpiketrain):
        pass