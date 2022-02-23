from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *

class Simulation():
    def __init__(self):
        self.spikeTrain = None
        self.liquid = Liquid()
        self.outputPop = OutputPopulation()
