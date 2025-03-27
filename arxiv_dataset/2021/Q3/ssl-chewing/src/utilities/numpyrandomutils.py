import numpy as np


def noise_psd(n: int, psd=lambda f: 1):
    y_white = np.fft.fft(np.random.randn(n))

    s = np.abs(np.fft.fftfreq(n))
    s = psd(s)
    s = s / np.sqrt(np.mean(s ** 2))

    y_colored = y_white * s

    x = np.fft.irfft(y_colored)
    x = np.real(x)

    return x


def noise_psd_generator(f):
    return lambda n: noise_psd(n, f)


@noise_psd_generator
def white_noise(f):
    return 1


@noise_psd_generator
def blue_noise(f):
    return np.sqrt(f)


@noise_psd_generator
def violet_noise(f):
    return f


@noise_psd_generator
def brownian_noise(f):
    return 1 / np.where(f == 0, float('inf'), f)


@noise_psd_generator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    plt.loglog(f, np.abs(np.fft.rfft(s)))


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    for G in [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise]:
        plot_spectrum(G(2**14))
    plt.legend(['brownian', 'pink', 'white', 'blue', 'violet'])
    plt.show()

    print('done')
