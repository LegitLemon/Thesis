import numpy as np
import matplotlib.pyplot as plt
from spikes import encoder
from scipy import signal
from brian2 import *
import LSM.neuronDynamics as nd
class SignalEncoder:
    def __init__(self):
        self.T = nd.amountOfSamples
        self.unspikedPatterns = []
        self.spikedPatterns = []
        self.initSpikedPatterns()
        self.offset = nd.offsetSignalEncoder
        indeces = len(self.spikedPatterns[0])*[0]
        self.spikeGenerator = SpikeGeneratorGroup(1, indeces, self.spikedPatterns[0]*(ms/10))

    def initUnspikedPatterns(self):
        # sine wave
        self.unspikedPatterns.append([np.sin(2 * n / (10 * np.sqrt(2))) for n in range(self.T)])
        # Square Wave
        self.unspikedPatterns.append([signal.square(n / 10) for n in range(self.T)])
        # regular cosine
        self.unspikedPatterns.append([np.cos(n / 10) for n in range(self.T)])
        # Sawtooth Wave
        self.unspikedPatterns.append([signal.sawtooth(n / 10) for n in range(self.T)])

    def initSpikedPatterns(self):
        self.initUnspikedPatterns()
        for unspikedPattern in self.unspikedPatterns:
            expandedPattern = np.expand_dims(np.asarray([unspikedPattern]), axis=2)
            expandedPatternSpikes = encoder.BSA(expandedPattern)
            spikeTimes = expandedPatternSpikes.get_spike_time()[0]
            self.spikedPatterns.append(spikeTimes)

    def plotEncodedSpikeSignal(self, index):
        y = [0 for i in range(len(self.spikedPatterns[index]))]
        normalizedSpikeTrain = [x/self.offset for x in self.spikedPatterns[index]]
        plt.plot(normalizedSpikeTrain, y, marker='o')
        plt.plot(self.unspikedPatterns[index])
        plt.show()

    def plotEncodedSpikesSignals(self):
        for num, (spikeTrain, analogSignal) in enumerate(zip(self.spikedPatterns, self.unspikedPatterns)):
            figure = plt.figure(num)
            y = [0 for i in range(len(spikeTrain))]
            plt.plot()
            normalizedSpikeTrain = [x/self.offset for x in spikeTrain]
            plt.plot(normalizedSpikeTrain, y, marker='o')
            plt.plot(analogSignal)
        plt.show()

    def updateInput(self, index):
        indeces = len(self.spikedPatterns[index])*[0]
        spikeTimes = self.spikedPatterns[index]*(ms/10)
        self.spikeGenerator.set_spikes(indeces, spikeTimes)

