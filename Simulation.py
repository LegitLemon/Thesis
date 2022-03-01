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

    def initInputSynapses(self):
        amount = int(0.3*nd.N_liquid)
        indeces = []
        for i in range(amount):
            index = random.randint(0, nd.N_liquid-1)
            while index in indeces:
                #print("hoi")
                index = random.randint(0, nd.N_liquid-1)
            indeces.append(index)
            weight = random.random()
            self.inputSynapses.connect(np.arrange(nd.poissonNum), index)
            self.inputSynapses.w[:, index] = weight

    def initOutputSynapses(self):
        print("Making input connections to LSM")
        self.inputSynapses.connect(np.arrange(nd.N_liquid), np.arrange(nd.N_output))
        self.inputSynapses.w[:, :] = 1

    def run(self):
        print("starting simulation")
        run(report=ProgressBar(), report_period=1*second)
        self.outputPop.getReadout(0)


