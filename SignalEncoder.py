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
        print(self.spikedPatterns[0])
        self.spikeGenerator = SpikeGeneratorGroup(1, indeces, self.spikedPatterns[0]*(ms))

    def initUnspikedPatterns(self):
        # # sine wave
        # self.unspikedPatterns.append([np.sin(2 * n / (10 * np.sqrt(2))) for n in range(self.T)])
        # # Square Wave
        # self.unspikedPatterns.append([signal.square(n / 10) for n in range(self.T)])
        # # regular cosine
        # self.unspikedPatterns.append([np.cos(n / 10) for n in range(self.T)])
        # # Sawtooth Wave
        # self.unspikedPatterns.append([signal.sawtooth(n / 10) for n in range(self.T)])
        stop = int(nd.simLength/ms)
        n = np.linspace(start=0, stop=5, num=stop)
        self.unspikedPatterns.append(np.sin(2*np.pi*n)+1)
        self.unspikedPatterns.append(signal.sawtooth(8*n)+1)
        self.unspikedPatterns.append(np.cos(2*np.pi*n)+1)
        self.unspikedPatterns.append(signal.square(4*n)+1)


    def filterSpikeTrainWithRefractory(self, spiketrain):
        normalizedSpiketrain = []
        count = 0
        prev = 0
        for spike in spiketrain:
            curr = spike
            count += curr-prev
            if count >= 15:
                count = 0
                normalizedSpiketrain.append(spike)
            prev = curr
        return normalizedSpiketrain

    def initSpikedPatterns(self):
        self.initUnspikedPatterns()
        for unspikedPattern in self.unspikedPatterns:
            expandedPattern = np.expand_dims(np.asarray([unspikedPattern]), axis=2)
            expandedPatternSpikes = encoder.BSA(expandedPattern, filter_length=6, cutoff=0.1, threshold=.6)
            spikeTimes = expandedPatternSpikes.get_spike_time(offset=1)[0]
            spikeTimes = self.filterSpikeTrainWithRefractory(spikeTimes)
            self.spikedPatterns.append(spikeTimes)

    def plotEncodedSpikeSignal(self, index):
        y = [0 for i in range(len(self.spikedPatterns[index]))]
        normalizedSpikeTrain = [x for x in self.spikedPatterns[index]]
        plt.plot(normalizedSpikeTrain, y, marker='|')
        plt.plot(self.unspikedPatterns[index])
        plt.show()

    def plotEncodedSpikesSignals(self):
        for num, (spikeTrain, analogSignal) in enumerate(zip(self.spikedPatterns, self.unspikedPatterns)):
            figure = plt.figure(num)
            x = self.filterSpikeTrainWithRefractory(spikeTrain)
            y = [0 for i in range(len(x))]
            print(len(spikeTrain), len(analogSignal))
            plt.plot()
            normalizedSpikeTrain = [j/1 for j in x]
            plt.plot(normalizedSpikeTrain, y, marker='|')
            plt.plot(analogSignal)
        plt.show()

    def updateInput(self, index):
        indeces = len(self.spikedPatterns[index])*[0]
        spikeTimes = self.spikedPatterns[index]*(ms)
        self.spikeGenerator.set_spikes(indeces, spikeTimes)

