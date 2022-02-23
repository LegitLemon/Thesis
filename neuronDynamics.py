# Parameters taken from (Maass, 2002)

from brian2 import *

N_liquid = 135
N_output = 20

# Neuron Parameters
refrac = 3 * ms
thres = "v>.15 mV"
reset = "v=13.5 mV"
tau = 30 * ms
delta = 5 * ms

eqs = '''
dv/dt = -(v-I)/tau: volt (unless refractory)
I = 15.5 nV: 1
'''