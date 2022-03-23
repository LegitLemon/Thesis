# Parameters taken from (Maass, 2002)

from brian2 import *

## Dimensionality of neurongroups
poissonNum = 1
N_liquid = 75
N_output = 25


amountOfPatternsClassifier = 1
amountOfRunsPerPattern = 1

###
connecProb = 0.7
proportionInhib = 0.4
proportionInputInjectionLiquid = 0.7

# simulation length in seconds
simLength = 5*second

# binsize in ms
binSize = 500*ms

# Neuron Parameters in liquid
refrac = 3 * ms
thres = "v>= 15 * mV"
reset = "v= 13.5 * mV"
tau = 30 * ms
delta = 5 * ms
weightEQ = "v += w"
eqs = '''
dv/dt = -(v)/(30*ms) + ((1*Mohm)*I)/(30*ms): volt (unless refractory)
I = 13.5 * nampere: ampere
'''

# Neuron parameters in output
refracOut = 0 * ms
thresOut = "v>= 15 * mV"
tauOut = 30 * ms
deltaOut = 5 * ms
weightEQOut = "v += w"
eqsOut = '''
dv/dt = -(v)/(30*ms) + ((1*Mohm)*I)/(30*ms): volt (unless refractory)
I = 13.5 * nampere: ampere
'''


## Parameters for setting connection
lam = 2

