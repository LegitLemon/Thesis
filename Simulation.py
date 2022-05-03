from LSM.Liquid import Liquid
from LSM.OutputPopulation import OutputPopulation
from brian2 import *
import LSM.neuronDynamics as nd
import random
import numpy as np
from Utils.ProgressBar import ProgressBar
import matplotlib.pyplot as plt
from brian2tools import *
from SignalEncoder import SignalEncoder

class Simulation:
    def __init__(self, control):
        self.signalEncoder = SignalEncoder()
        self.inputMonitor = SpikeMonitor(self.signalEncoder.spikeGenerator)
        self.liquid = Liquid(control)
        self.inputSynapses = Synapses(self.signalEncoder.spikeGenerator, self.liquid.liquid, model="w:volt", on_pre="v += w")
        self.initInputSynapses()

        self.outputPop = OutputPopulation()
        self.outputSynapses = Synapses(self.liquid.liquid, self.outputPop.outputPopulation, model="w:volt", on_pre="v += w")
        self.initOutputSynapses()
        self.network = Network()
        self.initNetwork()

    def initNetwork(self):
        print("Compiling network")
        #input
        self.network.add(self.signalEncoder.spikeGenerator)
        self.network.add(self.inputSynapses)
        self.network.add(self.inputMonitor)
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
            self.inputSynapses.w[:, index] = '0.7*rand()*mV'

    def initOutputSynapses(self):
        print("Making output connections to outputPopulation")
        for i in range(nd.N_liquid):
            self.outputSynapses.connect(i=i, j=np.arange(nd.N_output))
            self.outputSynapses.w[i, :] = '1.5*rand()*mV'

    def resetInput(self, index):
        self.signalEncoder.updateInput(index)

    def initClassifier(self):
        print("Starting Classification Procedure, initialising conceptors")
        for i in range(nd.amountOfPatternsClassifier):
            print("Running simulation on input pattern: ", i)
            self.computeStateMatrixClassification(i)
            self.resetInput(i)
        self.classifier.computeConceptors()
        print("Classifier Initialized")

    def plotRun(self, index):
        # plot0 = brian_plot(self.signalEncoder.spikeGenerator)
        plot0 = brian_plot(self.inputMonitor)
        self.savePlot(plot0, "Input")
        plt.show()
        self.signalEncoder.plotEncodedSpikeSignal(index)
        plot1 = brian_plot(self.outputPop.spikeMonitor)
        self.savePlot(plot1, "OutputResponse")
        plt.show()
        plot2 = brian_plot(self.liquid.spikemonitor)
        self.savePlot(plot2, "LiquidResponse")
        plt.show()
        plot3 = brian_plot(self.liquid.stateMonitor)
        self.savePlot(plot3, "LiquidState")
        plt.show()

    def savePlot(self, axisObject, name):
        plot = axisObject.get_figure()
        plot.savefig("Plots/"+name+".png")

    def run(self):
        print("starting simulation")
        self.liquid.reset()

        self.network.run(nd.simLength, report=ProgressBar(), report_period=0.2*second)
        self.plotRun(0)


