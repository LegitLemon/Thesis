from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *
import neuronDynamics as nd
import random
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt

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
        amount = int(0.3*nd.N_liquid)
        indeces = []
        for i in range(amount):
            index = random.randint(0, nd.N_liquid-1)
            while index in indeces:
                index = random.randint(0, nd.N_liquid-1)
            indeces.append(index)
            weight = random.random()
            self.inputSynapses.connect(j=index, i=np.arange(nd.poissonNum))
            self.inputSynapses.w[:, index] = weight * mV

    def initOutputSynapses(self):
        print("Making output connections to outputPopulation")
        for i in range(nd.N_liquid):
            self.outputSynapses.connect(i=i, j=np.arange(nd.N_output))
            if self.liquid.neurontypes[i] == False:
                self.outputSynapses.w[:, i] = 18 * nvolt
            else:
                self.outputSynapses.w[:, i] = 9 * nvolt


    def run(self):
        print("starting simulation")
        self.liquid.reset()
        self.network.run(1*second, report=ProgressBar(), report_period=0.2*second)

        fig1 = plt.figure(1)
        plt.plot(self.inputMonitor.t / ms, self.inputMonitor.i, '.k')

        fig2 = plt.figure(2)
        plt.plot(self.outputPop.spikeMonitor.t / ms, self.outputPop.spikeMonitor.i, 'k')
        plt.show()

        fig3 = plt.figure(3)
        for j in range(5):
            plt.plot(self.liquid.stateMonitor.t / ms, self.liquid.stateMonitor.v[j])
        plt.show()

        fig4 = plt.figure(4)
        for j in range(5):
            plt.plot(self.liquid.spikemonitor.t / ms, self.liquid.spikemonitor.i)
        plt.show()


