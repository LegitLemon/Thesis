# Parameters taken from (Maass, 2002)

from brian2 import *

N_liquid = 135
N_output = 55

# Neuron Parameters in liquid
refrac = 3 * ms
thres = "v>= 15 * mV"
reset = "v= 13.5 * mV"
tau = 30 * ms
delta = 5 * ms
weightEQ = "v += w"
eqs = '''
dv/dt = -(v-I)/(30*ms): volt (unless refractory)
I = 13.5 * nvolt: volt
'''

# Neuron parameters in output
refracOut = 0 * ms
thresOut = "v>= 15 * mV"
tauOut = 30 * ms
deltaOut = 5 * ms
weightEQOut = "v += w"
eqsOut = '''
dv/dt = -(v-I)/(30*ms): volt 
I = 13.5 * nvolt: volt
'''

## Parameters for setting connection
# lam = [2,4,8] in paper
lam = 2


##
binSize = 20


## Poisson Group simulation
poissonNum = 1
