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
