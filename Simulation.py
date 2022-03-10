from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *
import neuronDynamics as nd
import random
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
from Optimizer import Optimizer

class Simulation():
    def __init__(self):
        self.poissonInput = PoissonGroup(nd.poissonNum, np.arange(nd.poissonNum)*Hz + 10*Hz)
        self.inputMonitor = SpikeMonitor(self.poissonInput)
        self.liquid = Liquid()
        self.optim = Optimizer()
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
            self.inputSynapses.connect(i=np.arange(nd.poissonNum), j=index)
            self.inputSynapses.w[:, index] = 15 * mV

    def initOutputSynapses(self):
        print("Making output connections to outputPopulation")
        for i in range(nd.N_liquid):
            self.outputSynapses.connect(i=i, j=np.arange(nd.N_output))
            self.outputSynapses.w[i, :] = 2 * mV

    def run(self):
        print("starting simulation")
        self.liquid.reset()
        self.network.run(5*second, report=ProgressBar(), report_period=0.2*second)
        self.optim.stateMatrices.append(self.outputPop.getStateMatrix())

        fig1 = plt.figure(1)
        plt.plot(self.inputMonitor.t / ms, self.inputMonitor.i, '.k')
        plt.title(label="Input spike train")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="injected voltage")

        fig2 = plt.figure(2)
        plt.plot(self.outputPop.spikeMonitor.t / ms, self.outputPop.spikeMonitor.i, 'k')
        plt.title(label="output population spikes")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="output voltage")

        fig3 = plt.figure(3)
        for j in range(5):
            plt.plot(self.liquid.stateMonitor.t / ms, self.liquid.stateMonitor.v[j])
        plt.title(label="neuron voltage of select liquid neurons")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="membrane voltage")



        fig4 = plt.figure(4)
        for j in range(5):
            plt.plot(self.liquid.spikemonitor.t / ms, self.liquid.spikemonitor.i)
        plt.title(label="Spiking activity of select liquid neurons")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="output voltage")

        plt.show()


