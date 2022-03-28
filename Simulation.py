from LSM.Liquid import Liquid
from LSM.OutputPopulation import OutputPopulation
from brian2 import *
import LSM.neuronDynamics as nd
import random
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
from brian2tools import *
from Classification.Classifier import Classifier

class Simulation():
    def __init__(self):
        self.poissonInput = PoissonGroup(nd.poissonNum, np.arange(nd.poissonNum)*Hz + 10*Hz)
        self.inputMonitor = SpikeMonitor(self.poissonInput)
        self.liquid = Liquid()
        self.inputSynapses = Synapses(self.poissonInput, self.liquid.liquid, model="w:volt", on_pre="v += w")
        self.initInputSynapses()

        self.outputPop = OutputPopulation()
        self.outputSynapses = Synapses(self.liquid.liquid, self.outputPop.outputPopulation, model="w:volt", on_pre="v += w")
        self.initOutputSynapses()
        self.network = Network()
        self.initNetwork()

        self.classifier = Classifier()

    def initNetwork(self):
        print("Compiling network")
        #input
        self.network.add(self.poissonInput)
        self.network.add(self.inputMonitor)
        self.network.add(self.inputSynapses)

        #liquid
        self.network.add(self.liquid.liquid)
        self.network.add(self.liquid.synapses)
        self.network.add(self.liquid.stateMonitor)
        self.network.add(self.liquid.spikemonitor)

        #output
        self.network.add(self.outputPop.outputPopulation)
        self.network.add(self.outputPop.spikeMonitor)
        self.network.add(self.outputSynapses)


    def initInputSynapses(self):
        print("making Connection to input LSM")
        amount = int(nd.proportionInputInjectionLiquid*nd.N_liquid)
        indeces = []
        for i in range(amount):
            index = random.randint(0, nd.N_liquid-1)
            while index in indeces:
                index = random.randint(0, nd.N_liquid-1)
            indeces.append(index)
            self.inputSynapses.connect(i=np.arange(nd.poissonNum), j=index)
            self.inputSynapses.w[:, index] = '1.5*rand()*mV'

    def initOutputSynapses(self):
        print("Making output connections to outputPopulation")
        for i in range(nd.N_liquid):
            self.outputSynapses.connect(i=i, j=np.arange(nd.N_output))
            self.outputSynapses.w[i, :] = '1.5*rand()*mV'

    def resetInput(self):
        self.classifier.inputSpikeTrains.append((self.poissonInput, self.inputMonitor))
        self.network.remove(self.inputMonitor)
        self.network.remove(self.poissonInput)
        self.poissonInput = PoissonGroup(nd.poissonNum, np.arange(nd.poissonNum) * Hz + 10 * Hz)
        self.inputMonitor = SpikeMonitor(self.poissonInput)
        self.network.add(self.poissonInput)
        self.network.add(self.inputMonitor)

    def initClassifier(self):
        print("Starting Classification Procedure, initialising conceptors")
        for i in range(nd.amountOfPatternsClassifier):
            print("Running simulation on input pattern: ", i)
            self.computeStateMatrix()
            # self.resetInput()
        self.classifier.computeConceptors()
        print("Classifier Initialized")

    def testClassifier(self):
        pass

    def computeStateMatrix(self):
        stateVectors = []
        for i in range(nd.amountOfRunsPerPattern):
            self.liquid.reset()
            self.network.run(nd.simLength)
            stateVector = self.liquid.computeStateMatrix(self.inputMonitor.t)
            stateVectors.append(stateVector)
            self.plotRun()
            # self.network.restore()
        stateMatrix = np.asarray(stateVectors[0])
        print(stateMatrix.shape)

        self.classifier.stateMatrices.append(stateMatrix)


    def plotRun(self):
        fig1 = plt.figure(1)
        plt.plot(self.inputMonitor.t / ms, self.inputMonitor.i, '.k')
        plt.title(label="Input spike train")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="injected voltage")
        plt.show()

        plot1 = brian_plot(self.outputPop.spikeMonitor)
        plt.show()
        plot2 = brian_plot(self.liquid.spikemonitor)
        plt.show()

        plot3 = brian_plot(self.liquid.stateMonitor)
        plt.show()

    def run(self):
        print("starting simulation")
        self.liquid.reset()
        self.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
        self.plotRun()


