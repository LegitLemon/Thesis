from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *
import neuronDynamics as nd
import random
import numpy as np
from ProgressBar import ProgressBar

class Simulation():
    def __init__(self):
        self.poissonInput = PoissonGroup(nd.poissonNum, np.arange(nd.poissonNum)*Hz + 10*Hz)
        self.liquid = Liquid()
        self.inputSynapses = Synapses(self.poissonInput, self.liquid.liquid, model="w:volt", on_pre="v += w")
        self.initInputSynapses()

        self.outputPop = OutputPopulation()
        self.outputSynapses = Synapses(self.liquid.liquid, self.outputPop.outputPopulation, model="w:volt", on_pre="v += w")
        self.initOutputSynapses()
        self.network = Network()
        self.initNetwork()

    def initNetwork(self):
        self.network.add(self.poissonInput)
        self.network.add(self.liquid.liquid)
        self.network.add(self.inputSynapses)
        self.network.add(self.liquid.synapses)
        self.network.add(self.outputPop.outputPopulation)
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
        self.outputSynapses.w[:, :] = 1 * mV

    def run(self):
        print("starting simulation")
        self.network.run(3*second, report=ProgressBar(), report_period=1*second)
        self.outputPop.getReadout(0)


