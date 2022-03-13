from Liquid import Liquid
from OutputPopulation import OutputPopulation
from brian2 import *
import neuronDynamics as nd
import random
import numpy as np
from ProgressBar import ProgressBar
import matplotlib.pyplot as plt
from brian2tools import *

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
            self.inputSynapses.connect(i=np.arange(nd.poissonNum), j=index)
            self.inputSynapses.w[:, index] = nd.inputLiquidSynapseStrength

    def initOutputSynapses(self):
        print("Making output connections to outputPopulation")
        for i in range(nd.N_liquid):
            self.outputSynapses.connect(i=i, j=np.arange(nd.N_output))
            self.outputSynapses.w[i, :] = nd.liquidOutputSynapseStrength

    def rasterPLot(self, spike_monitor):
        brian_plot(spike_monitor)

    def visualise_connectivity(self, S):
        Ns = len(S.source)
        Nt = len(S.target)
        figure(figsize=(10, 4))
        subplot(121)
        plot(zeros(Ns), arange(Ns), 'ok', ms=10)
        plot(ones(Nt), arange(Nt), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plot([0, 1], [i, j], '-k')
        xticks([0, 1], ['Source', 'Target'])
        ylabel('Neuron index')
        xlim(-0.1, 1.1)
        ylim(-1, max(Ns, Nt))
        subplot(122)
        plot(S.i, S.j, 'ok')
        xlim(-1, Ns)
        ylim(-1, Nt)
        xlabel('Source neuron index')
        ylabel('Target neuron index')

    def run(self):
        self.visualise_connectivity(self.liquid.synapses)
        print("starting simulation")
        self.liquid.reset()
        self.network.run(1*second, report=ProgressBar(), report_period=0.2*second)
        fig1 = plt.figure(1)
        plt.plot(self.inputMonitor.t / ms, self.inputMonitor.i, '.k')
        plt.title(label="Input spike train")
        plt.xlabel(xlabel="time in s")
        plt.ylabel(ylabel="injected voltage")
        plt.show()

        plot1 = brian_plot(self.outputPop.spikeMonitor)
        plt.show()
        plot2 = brian_plot(self.liquid.spikemonitor)
        plt.show()

        plot3 = brian_plot(self.liquid.stateMonitor)
        plt.show()


        # fig2 = plt.figure(2)
        # plt.plot(self.outputPop.spikeMonitor.t / ms, self.outputPop.spikeMonitor.i, 'k')
        # plt.title(label="output population spikes")
        # plt.xlabel(xlabel="time in s")
        # plt.ylabel(ylabel="output voltage")

        # fig3 = plt.figure(3)
        # for j in range(5):
        #     plt.plot(self.liquid.stateMonitor.t / ms, self.liquid.stateMonitor.v[j])
        # plt.title(label="neuron voltage of select liquid neurons")
        # plt.xlabel(xlabel="time in s")
        # plt.ylabel(ylabel="membrane voltage")



        # fig4 = plt.figure(4)
        # for j in range(5):
        #     plt.plot(self.liquid.spikemonitor.t / ms, self.liquid.spikemonitor.i)
        # plt.title(label="Spiking activity of select liquid neurons")
        # plt.xlabel(xlabel="time in s")
        # plt.ylabel(ylabel="output voltage")
        # plt.show()


