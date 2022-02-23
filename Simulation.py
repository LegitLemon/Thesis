from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *

class Simulation():
    def __init__(self):
        self.spikeTrain = None
        self.liquid = Liquid()
        self.outputPop = OutputPopulation()
        self.outputSynapses = Synapses(self.liquid.liquid, self.outputPop.outputPopulation, model="w:volt", on_pre="v += w")