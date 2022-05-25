import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def single_leaky_esn(x, t, tau, a, w_connec, w_input):
    decay = -a * x
    new_state = np.tanh(w_input*np.sin(3*t)+w_connec*x)
    dxdt = (1/tau)*(decay + new_state)
    return dxdt

tau = 2
a = 0.98
w_connec = 0.5
w_input = 0.5

#
# parameters = (tau, a, w_connec, w_input)

x0 = 0.5
sampling_frequency = 10
time = 30
t = np.linspace(0, time, num=time*sampling_frequency)

# solve ODE
results = []
a = np.linspace(0.4, 1, num=5)

for i in a:
    parameters = (tau, i, w_connec, w_input)
    y = odeint(single_leaky_esn, x0, t, parameters)
    plt.plot(t, y, label=i)

# plot results
plt.xlabel('time')
plt.ylabel('x(t)')
plt.title("effect of leaking rate on neuron dynamics")
plt.legend()
plt.show()


tau = np.linspace(1, 10, num=5)
leak = 0.98
for i in tau:
    parameters = (i, leak, w_connec, w_input)
    y = odeint(single_leaky_esn, x0, t, parameters)
    plt.plot(t, y, label=i)

# plot results
plt.xlabel('time')
plt.title("effect of time constant on neuron dynamics")
plt.ylabel('x(t)')
plt.legend()
plt.show()


