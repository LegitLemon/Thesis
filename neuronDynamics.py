from brian2 import *

N_liquid = 100
N_output = 20

# Neuron Parameters
refrac = 5 * ms
thres = "v>0.2"
reset = "v=0"
tau = 5 * ms
delta = 5 * ms
eqs = '''
dv/dt = -(v-I)/tau: 1 (unless refractory)
I = sin(t/delta) : 1
'''