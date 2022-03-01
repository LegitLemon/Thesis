# Parameters taken from (Maass, 2002)

from brian2 import *

N_liquid = 135
N_output = 51

# Neuron Parameters in liquid
refrac = 3 * ms
thres = "v>= .15 * mV"
reset = "v= 13.5 * mV"
tau = 30 * ms
delta = 5 * ms
weightEQ = "v += w"
eqs = '''
dv/dt = -(v-I)/(30*ms): volt (unless refractory)
I = 0.000155 * mV: volt
'''

# Neuron parameters in output
refracOut = 3 * ms
thresOut = "v>= .15 * mV"
resetOut = "v=13.5 * mV"
tauOut = 30 * ms
deltaOut = 5 * ms
weightEQOut = "v += w"
eqsOut = '''
dv/dt = -(v-I)/(30*ms): volt (unless refractory)
I = 0.000135 * mV: volt
'''

## Parameters for setting connection
# lam = [2,4,8] in paper
lam = 2


##
binSize = 20


## Poisson Group simulation
poissonNum = 100
