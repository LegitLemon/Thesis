# class representing the liquid in the LSM implementation
import random

from brian2 import *
import neuronDynamics as nd
import random

class Liquid():

    def __init__(self):
        print("Making new LSM")
        self.neurontypes = self.initNeuronTypes()
        self.liquid = NeuronGroup(N=nd.N_liquid, threshold=nd.thres, model=nd.eqs, refractory=nd.refrac, reset=nd.reset)
        self.synapses = Synapses(self.liquid, self.liquid, model="w:volt", on_pre=nd.weightEQ)
        print("starting LSM synapses")
        self.initSynapses()
        print("connected LSM synapses")

    def initNeuronTypes(self):
        # initialise all neurons to be excitatory
        neuronTypes = [False for x in range(nd.N_liquid)]
        # compute the amount of inhibitory connections
        amount = int(0.2*nd.N_liquid)
        # Set that amount of neurons to inhibitory
        indeces = []
        for x in range(amount):
            ind = random.randint(0, nd.N_liquid-1)

            while ind in indeces:
                ind = random.randint(0, nd.N_liquid-1)
            indeces.append(ind)
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
            value = random.random()
            if self.neurontypes[neuronFrom] == False:
                value *= -1
            self.synapses.connect(i=neuronFrom, j=neuronTo)
            self.synapses.w[neuronFrom, neuronTo] = value * mV

    def reset(self):
        self.liquid.v = "random.uniform(13,5 ,15) * mV"
