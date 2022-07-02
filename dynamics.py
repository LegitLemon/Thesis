from networkParameters import *

control_constant = 50

def input1(t):
    return 2*np.sin(t/20)

def input2(t):
    return 4*np.sin((t/10) + 0.5*np.pi)


def get_input(number, t):
    if number == 1:
        return input1(t)
    else:
        return input2(t)


def leaky_esn_conceptor_projection(state, t, tau, a_input, w_connec, conceptor, P):
    x = state[:N]

    x_bar = np.dot(conceptor, x)

    decay_input = np.dot(-a_input, x_bar)
    new_state_input = np.tanh(np.dot(w_connec, x_bar))

    control_term = np.dot(P, x)
    control_errors_projection.append(control_term)
    control_term *= control_constant

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_projection.append(dist)

    dxdt = (1/tau)*(decay_input+new_state_input - control_term)

    return dxdt

def leaky_esn_conceptor_negation(state, t, tau, a_input, w_connec, conceptor):
    x = state[:N]

    x_bar = np.dot(conceptor, x)


    decay_input = np.dot(-a_input, x_bar)
    new_state_input = np.tanh(np.dot(w_connec, x_bar))

    control_term = np.dot((np.identity(N)-conceptor), x)
    control_errors_negation.append(control_term)
    control_term *= control_constant

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_negation.append(dist)

    dxdt = (1/tau)*(decay_input + new_state_input - control_term)

    return dxdt


def leaky_esn_no_control(state, t, tau, a_input, w_connec, conceptor):
    x = state[:N]
    x_bar = np.dot(conceptor, x)

    decay_input = np.dot(-a_input, x_bar)
    new_state_input = np.tanh(np.dot(w_connec, x_bar))

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_no_control.append(dist)

    dxdt = (1/tau)*(decay_input + new_state_input)

    return dxdt


def leaky_esn(state, t, tau, a_input, w_connec, w_input, input_number):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    input_driven = np.dot(w_input, get_input(input_number, t))
    old_state_input = np.dot(w_connec, x)

    new_state_input = np.tanh(input_driven + old_state_input)

    dxdt = (1/tau)*(decay_input + new_state_input)

    return dxdt


def test_loading(state, tau, a_input, w_connec):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    old_state_input = np.dot(w_connec, x)

    new_state_input = np.tanh(old_state_input)

    dxdt = (1/tau)*(decay_input + new_state_input)

    return dxdt


def rossler(state, t, a=0.2, b=0.2, c=8):
    [x, y, z] = state
    x_dot = -1*(y+z)
    y_dot = x + a*y
    z_dot = b + x*z -c*z
    return [x_dot, y_dot, z_dot]


# Lorenz attractor
def lorenz(state, t, s=10, r=28, b=2.667):
    [x, y, z] = state
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return [x_dot, y_dot, z_dot]


def mackey_glass(length=len(t), x0=None, a=0.2, b=0.1, c=10.0, tau=17.0,
                 n=1000, sample=0.46, discard=250):
    """
    x0 : array, optional (default = random)
        Initial condition for the discrete map.  Should be of length n.
    a : float, optional (default = 0.2)
        Constant a in the Mackey-Glass equation.
    b : float, optional (default = 0.1)
        Constant b in the Mackey-Glass equation.
    c : float, optional (default = 10.0)
        Constant c in the Mackey-Glass equation.
    tau : float, optional (default = 23.0)
        Time delay in the Mackey-Glass equation.
    n : int, optional (default = 1000)
        The number of discrete steps into which the interval between
        t and t + tau should be divided.  This results in a time
        step of tau/n and an n + 1 dimensional map.
    sample : float, optional (default = 0.46)
        Sampling step of the time series.  It is useful to pick
        something between tau/100 and tau/10, with tau/sample being
        a factor of n.  This will make sure that there are only whole
        number indices.
    discard : int, optional (default = 250)
        Number of n-steps to discard in order to eliminate transients.
        A total of n*discard steps will be discarded.
    """
    sample = int(n * sample / tau)
    grids = n * discard + sample * length
    x = np.empty(grids)

    if not x0:
        x[:n] = 0.5 + 0.05 * (-1 + 2 * np.random.random(n))
    else:
        x[:n] = x0

    A = (2 * n - b * tau) / (2 * n + b * tau)
    B = a * tau / (2 * n + b * tau)

    for i in range(n - 1, grids - 1):
        x[i + 1] = A * x[i] + B * (x[i - n] / (1 + x[i - n] ** c) +
                                   x[i - n + 1] / (1 + x[i - n + 1] ** c))
    return x[n * discard::sample]


