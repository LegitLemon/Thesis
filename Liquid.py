# class representing the liquid in the LSM implementation
import random

from brian2 import *
import neuronDynamics as nd
import random

class Liquid():

    def __init__(self):
        self.neurontypes = self.initNeuronTypes()
        self.liquid = NeuronGroup(N=nd.N_liquid, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)
        self.synapses = Synapses(self.liquid, self.liquid, on_pre=nd.weightEQ)
        self.initSynapses()

    def initNeuronTypes(self):
        # initialise all neurons to be excitatory
        neuronTypes = [False for x in range(len(nd.N_liquid))]
        # compute the amount of inhibitory connections
        amount = int(0.2*nd.N_liquid)
        # Set that amount of neurons to inhibitory
        for x in range(amount):
            ind = random.randint(0, nd.N_liquid)
            neuronTypes[ind] = True
        return neuronTypes

    # Euclidian norm on R^{3}
    def getDistance(self, a, b):
        diff = 0
        for i in range(len(a)):
            diff += (a[i]-b[i])**2
        return sqrt(diff)

    def initSynapses(self):
        number = 0
        for z in range(9):
            for y in range(3):
                for x in range(3):
                    number += 1
                    self.initNeuron((x, y, z), number)

    def initNeuron(self, p1, neuronFrom):
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

        if self.neurontypes[neuronFrom] is True:
            if self.neurontypes[neuronTo] is True:
            # EE
                C = 0.3
            else:
            # EI
                C = 0.2
        else:
            if self.neurontypes[neuronTo] is True:
            #IE
                C = 0.4
            else:
            #II
                C = 0.1

        prob = C * exp(exponent)
        if random.random() < prob:
            if self.neurontypes[neuronFrom] == True:
                value = random.random()
                while(value <0):
                    value = random.random()
                self.synapses.connect(neuronFrom, neuronTo)
                self.synapses[neuronFrom, neuronTo] = value
            else:
                value = random.random()
                while(value > 0):
                    value = random.random()
                self.synapses.connect(neuronFrom, neuronTo)
                self.synapses[neuronFrom, neuronTo] = value

    def getLiquidState(self):
        pass