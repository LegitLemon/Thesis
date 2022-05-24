# Parameters taken from (Maass, 2002)

from brian2 import *

## Dimensionality of neurongroups
poissonNum = 1
N_liquid = 50
N_output = 1


# binsize in ms
binSize = 25*ms
# The amount of binned windows to be discarded when constructing
washoutTime = 2
# simulation length in seconds
simLength = 5*second
amountOfBins = int(simLength/binSize)
offsetSignalEncoder = 100
amountOfSamples = int(simLength/(offsetSignalEncoder*(ms/10)))

amountOfPatternsClassifier = 4
amountOfRunsPerPattern = 2

lowerboundThreshold = 14.5 * mV
upperboundThreshold = 17 * mV

###
connecProb = 0.5
proportionInhib = 0.3
proportionInputInjectionLiquid = 1

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

eqsDynamicThreshold = '''
dv/dt = -(v)/(tau) + ((1*Mohm)*I)/(tau): volt (unless refractory)
tau : second
I = 13.5 * nampere: ampere
v_th : volt  # neuron-specific threshold
'''
thresOutDynamicThreshold = "v >= v_th"
resetDynamicThreshold = "v = 0.8*v_th"

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

