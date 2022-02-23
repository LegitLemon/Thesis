# class representing the liquid in the LSM implementation
import random

from brian2 import *
import neuronDynamics as nd
import random

class Liquid():

    def __init__(self):
        self.liquid = NeuronGroup(N=nd.N_liquid, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)
        self.synapses = Synapses(self.liquid, self.liquid, on_pre=nd.weightEQ)
        self.initSynapses()

    # Euclidian norm on R^{3}
    def getDistance(self, a, b):
        return 1

    def initSynapses(self):
        number = 0
        for z in range(9):
            for y in range(3):
                for x in range(3):
                    number += 1
                    self.initNeuron((x, y, z), number)

    def initNeuron(self, neuronFrom, p1):
        neuronTo = 0
        for z in range(9):
            for y in range(3):
                for x in range(3):
                    neuronTo += 1
                    dist = self.getDistance(p1, (x, y, z))
                    self.setConncetion(dist, neuronFrom, neuronTo)

    def setConncetion(self, dist, neuronFrom, neuronTo):
        # Probabillity of a connection between a and b C*e^-(D(a,b)/lambda)^(2)
        exponent = -1*(dist/nd.lam)
        exponent = exponent ** 2
        prob = nd.C * exp(exponent)
        if random.random() < prob:
            value = random.random()
            self.synapses.connect(neuronFrom, neuronTo)
            self.synapses[neuronFrom, neuronTo] = value

    def getLiquidState(self):
        pass