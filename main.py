from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# Connect Reservoir via weights
def initWeights(S, N):
    for i in range(N):
        for j in range(N):
            S.connect(i=i, j=j)
            S.w[i, j] = float(np.random.normal(0, 1))

def main():
    # Neuron Parameters
    refrac = 5*ms
    thres = "v>0.2"
    reset = "v=0"
    tau = 5*ms
    delta = 5*ms
    eqs = '''
    dv/dt = -(v-I)/tau: 1 (unless refractory)
    I = sin(t/delta) : 1
    '''

    # Simulation Parameters
    T = 0.5*second
    N = 50
    G = NeuronGroup(N, eqs, reset=reset, threshold=thres, refractory=refrac)
    statemon_neuron = StateMonitor(G, 'v', record=0)
    statemon_input = StateMonitor(G, 'I', record=0)
    spikemon = SpikeMonitor(G)

    # Make synapses between neurons
    weight_eq = "w : 1"
    impuls_eq = "v_post += w"
    S = Synapses(G, G, weight_eq, on_pre=impuls_eq)
    initWeights(S, N)
    run(T)

    # Analytics trackers
    plt.plot(statemon_neuron.t, statemon_neuron.v[0], label="spiking neuron")
    plt.plot(statemon_input.t, statemon_input.I[0], label="output")
    plt.show()


if __name__ == "__main__":
    main()