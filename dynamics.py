from networkParameters import *

def leaky_esn_conceptor_projection(state, t, tau, a_input, w_connec, conceptor, P):
    x = state[:N]

    # decay_input = np.dot(np.dot(-a_input, conceptor), x)
    decay_input = np.dot(-a_input, x)

    # old_state_input = np.dot(np.dot(w_connec, conceptor), x)

    new_state_input = np.tanh(np.dot(w_connec, x))
    conceptorProjection = np.dot(conceptor, (decay_input+new_state_input))
    # new_state_input = np.tanh(old_state_input)

    control_term = 0.5*np.dot(P, x)
    control_errors_projection.append(control_term)

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_projection.append(dist)

    dxdt = (1/tau)*(conceptorProjection - control_term)

    return dxdt

def leaky_esn_conceptor_negation(state, t, tau, a_input, w_connec, conceptor, P):
    x = state[:N]

    decay_input = np.dot(-a_input, x)
    new_state_input = np.tanh(np.dot(w_connec, x))
    conceptorProjection = np.dot(conceptor, (decay_input+new_state_input))

    control_term = np.dot((np.identity(N)-conceptor), x)
    control_errors_negation.append(control_term)

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_negation.append(dist)

    dxdt = (1/tau)*(conceptorProjection - control_term)

    return dxdt


def leaky_esn_no_control(state, t, tau, a_input, w_connec, conceptor, P):
    x = state[:N]
    decay_input = np.dot(-a_input, x)
    new_state_input = np.tanh(np.dot(w_connec, x))
    conceptorProjection = np.dot(conceptor, (decay_input+new_state_input))

    dist = np.linalg.norm(np.dot(conceptor, x) - x)
    conceptor_distance_no_control.append(dist)

    dxdt = (1/tau)*(conceptorProjection)

    return dxdt




def leaky_esn(state, t, tau, a_input, w_connec, w_input, w_output, bias):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    input_driven = np.dot(w_input, 2*np.sin(t/20))
    old_state_input = np.dot(w_connec, x)

    new_state_input = np.tanh(input_driven + old_state_input)

    dxdt = (1/tau)*(decay_input + new_state_input)

    return dxdt
