import LSM.neuronDynamics as nd
from Classification.Conceptor import Conceptor
from Simulation import Simulation
from brian2 import *
from Utils.ProgressBar import ProgressBar
import builtins as builtins

class Classifier:
    def __init__(self):
        self.alpha = 0.5
        self.stateMatrices = []
        self.inputSpikeTrains = []
        self.conceptors = []
        self.correlations = []
        self.simulation = Simulation()

    def computeConceptors(self):
        print("Computing Conceptors")
        for stateMatrix in self.stateMatrices:
            R = np.dot(stateMatrix, stateMatrix.transpose()) / nd.amountOfRunsPerPattern
            self.conceptors.append(Conceptor(R, self.alpha, nd.N_liquid))

    def computePositiveEvidence(self, z):
        evidences = []
        for C in self.conceptors:
            evidences.append(np.dot(np.dot(z.transpose(), C.C), z))
        return evidences

    def classify(self, z):
        evidences = self.computePositiveEvidence(z)
        print("Prediction evidences: ", evidences)
        judgement = np.argmax(np.array(evidences))
        print("Judgement: ", judgement)
        return judgement


    def initClassifier(self):
        print("Starting Classification Procedure, initialising conceptors")
        for i in range(nd.amountOfPatternsClassifier):
            print("Running simulation on input pattern: ", i)
            self.computeStateMatrixClassification(i)
            self.simulation.resetInput(i)
        self.computeConceptors()
        print("Classifier Initialized")

    def testClassifier(self):
        for i in range(len(self.conceptors)):
            self.simulation.liquid.reset()
            self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
            stateVector = self.simulation.liquid.computeStateMatrixClassification(self.simulation.signalEncoder.spikedPatterns[i] * ms)
            self.simulation.network.restore()
            self.classify(np.array(stateVector))
            self.simulation.resetInput(i)

        correct = 0
        false = 0
        howManyTests = builtins.input()
        print("How many tests would you like to do?")
        for i in range(int(howManyTests)):
            scaling, leftOrRight, testSignal, answer = self.simulation.signalEncoder.permutateSignal()
            spikePermutatedSignal = self.simulation.signalEncoder.initSpikePattern(testSignal)
            indeces = [0]*len(spikePermutatedSignal)
            spikeTimes = spikePermutatedSignal*ms
            self.simulation.signalEncoder.spikeGenerator.set_spikes(indeces, spikeTimes)
            self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
            stateVector = self.simulation.liquid.computeStateMatrixClassification(spikeTimes)
            judgement = self.classify(np.array(stateVector))

            if (judgement != answer):
                false += 1
                print("Incorrect classification")
                print("Correct answer was: ",
                      answer)
            else:
                correct += 1
                print("Correct classification")
            self.simulation.network.restore()
        self.printClassifierStatistics(correct, false)


    def printClassifierStatistics(self, correct, false):
        total = correct + false
        print("percentage of jedgements correct: ", correct/total*100, "%")
        print("total: ", total)
        print("correct: ", correct, "/", total)
        print("incorrect: ", false, "/", total)

    def computeStateMatrixClassification(self, index):
        stateVectors = []
        self.simulation.network.store()
        for i in range(nd.amountOfRunsPerPattern):
            self.simulation.liquid.reset()
            self.simulation.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
            stateVector = self.simulation.liquid.computeStateMatrixClassification(self.simulation.signalEncoder.spikedPatterns[index] * ms)
            stateVectors.append(stateVector)
            # self.plotRun(index)
            self.simulation.network.restore()

        stateMatrix = np.asarray(stateVectors).transpose()
        print("shape state matrix: ", stateMatrix.shape)

        self.stateMatrices.append(stateMatrix)

