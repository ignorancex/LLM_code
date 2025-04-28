import numpy as np
from matplotlib import pyplot as plt


def main():
    sigma = np.linspace(0.667, 1 / 0.35, int(1e4))
    coefficients = np.array(
        [
            -5.07531869,
            15.40598431,
            -16.45325227,
            8.68203863,
            -2.25753936,
            0.22934861
        ]
    )[::-1]
    order = 9

    polynomial = np.poly1d(coefficients)
    reflectivity = polynomial(sigma) ** order
    reflectivity_fft = np.fft.fft(reflectivity, norm="ortho")
    frequencies = np.fft.fftfreq(len(reflectivity), (sigma[1] - sigma[0]))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sigma, reflectivity, label=str(polynomial))
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Polynomial Plot')
    plt.legend()
    plt.grid(True)
    plt.ylim([-0.1, 1.1])

    plt.subplot(1, 2, 2)
    plt.plot(frequencies, np.abs(reflectivity_fft))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT of Polynomial')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
