import numpy as np

def generate_multisine(nu, N, fmin, fmax, umin, umax, nf):
    if nu == 1:
        umin_loc = np.array([umin])
        umax_loc = np.array([umax])
    oversampling_factor = 10 
    fs = oversampling_factor * fmax  # Sampling frequency in Hz
    dt = 1 / fs  # Sampling interval
    T = N * dt 
    t = np.linspace(0, T, N, endpoint=False)

    U = np.zeros((N, nu))

    for i in range(nu):
        frequencies = np.random.uniform(fmin, fmax, nf)
        phases = np.random.uniform(0, 2 * np.pi, nf)
        u = np.zeros(N)
        for freq, phase in zip(frequencies, phases):
            u += np.sin(2 * np.pi * freq * t + phase)
        u /= np.max(np.abs(u))
        u = u * (umax_loc[i] - umin_loc[i]) / 2 + (umax_loc[i] + umin_loc[i]) / 2
        U[:, i] = u

    return U
