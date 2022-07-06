from networkParameters import *
from computation import *
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
    [x, y, z] =  state
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return [x_dot, y_dot, z_dot]


def mackey_glass(length=len(t)+18, x0=None, a=0.2, b=0.1, c=10.0, tau=17.0,
                 n=1000, sample=0.46, discard=250):
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


def couple_pairs(data, tau=17):
    x, y = [], []
    for idx, dat in enumerate(data):
        if idx > 17:
            x.append(dat)
            y.append(data[idx-tau])
    return x, y

def generate_mg_data():
    mackey_glass_data = mackey_glass()
    x_data, z_data = couple_pairs(mackey_glass_data)
    x_data = normalise_time_series(x_data)
    z_data = normalise_time_series(z_data)
    input = []
    for idx, point in enumerate(x_data):
        input.append([x_data[idx], z_data[idx]])

    data_dict = {}
    for idx, point in enumerate(t):
        data_dict[str(point)] = input[idx]

    return data_dict

def generate_pattern_matrix_from_dict(attractor_dict):
    input = [attractor_dict[key] for key in attractor_dict.keys()]
    input = np.array(input).T 
    return input[:,washout_time:]

def generate_attractor_data(initial_condition=[0,1,1], attractor=lorenz, index1=0, index2=2):
    parameters = (t)
    data_lorenz = odeint(attractor, initial_condition, parameters)
    print(np.array([data_lorenz.shape]))
    x_data = [x[index1] for x in data_lorenz]
    z_data = [z[index2] for z in data_lorenz]
    x_data = normalise_time_series(x_data)
    z_data = normalise_time_series(z_data)

    input = []
    for idx, point in enumerate(x_data):
        input.append([x_data[idx], z_data[idx]])

    data_dict = {}
    for idx, point in enumerate(t):
        data_dict[str(point)] = input[idx]

    return data_dict


lorenz_input_series = generate_attractor_data()
rossler_input_series = generate_attractor_data([1, 1, 0], rossler, 0, 1)
mg_input_series = generate_mg_data()

def get_attractor_input(t, input_number):
    temp_data = None 
    if input_number == 0:
        temp_data = lorenz_input_series
    if input_number == 1:
        temp_data = rossler_input_series
    if input_number == 2:
        temp_data = mg_input_series

    val = temp_data.get(t)
    if val is None:
        for time in temp_data.keys():
            if float(time) > t:
                val = temp_data.get(time)
                break
    if val is None:
        val = get_attractor_input(t-0.1, input_number)
    return val


def leaky_esn_two_inputs(state, t, tau, a_input, w_connec, w_input, input_number):
    x = state[:N]
    decay_input = np.dot(-a_input, x)

    input_driven = np.dot(w_input, get_attractor_input(t, input_number))

    old_state_input = np.dot(w_connec, x)

    new_state_input = np.tanh(input_driven + old_state_input)

    dxdt = (1/tau)*(decay_input + new_state_input)


    return dxdt
def leaky_esn_two_inputs_control(state, t, tau, a_input, w_connec, w_input, input_number, conceptor, control_errors, control_distance):
    x = state[:N]
    x_bar = np.dot(conceptor, x)
    decay_input = np.dot(-a_input, x)

    input_driven = np.dot(w_input, get_attractor_input(t, input_number))

    old_state_input = np.dot(w_connec, x_bar)

    new_state_input = np.tanh(input_driven + old_state_input)
    control_term = np.dot((np.identity(N)-conceptor), x)

    control_errors[input_number].append(control_term)
    control_distance[input_number].append(np.linalg.norm(x_bar-x))

    dxdt = (1/tau)*(decay_input + new_state_input - 2.5*control_term)


    return dxdt