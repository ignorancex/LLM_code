import elasticdeform
import numpy as np
from numpy import ndarray
from scipy.ndimage import affine_transform


def brightness(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Applies a random brightness augmentation to the x component.

    :param x:
    :param y:
    """
    x_new = np.zeros(x.shape)
    for i in range(x.shape[-1]):
        im = x[:, :, :, i]
        gain = np.random.uniform(0.8, 2.3)
        gamma = np.random.uniform(0.8, 2.3)
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        x_new[:, :, :, i] = im_new

    return x_new, y


def elastic(x: ndarray, y: ndarray, sigma: tuple[float, float]) -> tuple[ndarray, ndarray]:
    """
    Applies elastic deformation to the X and Y equally. The severity is determined by the value of sigma.

    :param x:
    :param y:
    :param sigma:
    """
    [x_el, y_el] = elasticdeform.deform_random_grid(
        [x, y], sigma=np.random.uniform(sigma[0], sigma[1]), axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0],
        mode='constant')

    return x_el, y_el


def rotation(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees.

    :param x:
    :param y:
    :return:
    """
    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 10
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    R_rot = np.dot(np.dot(Rx, Ry), Rz)

    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0],
                        [0, beta, 0],
                        [0, 0, gamma]])

    R = np.dot(R_rot, R_scale)
    X_rot = np.empty_like(x)
    y_rot = np.empty_like(y)
    for b in range(x.shape[0]):
        for channel in range(x.shape[-1]):
            X_rot[b, :, :, :, channel] = affine_transform(
                x[b, :, :, :, channel], R, offset=0, order=1, mode='constant')
    for b in range(x.shape[0]):
        y_rot[b, :, :, :] = affine_transform(
            y[b, :, :, :], R, offset=0, order=0, mode='constant')

    return X_rot, y_rot


def combine_aug(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Combines the brightness and elastic deformation augmentations with a 30% of each augmentation being applied.

    :param x:
    :param y:
    """
    x_new, y_new = x, y

    if np.random.randint(0, 10) < 3:
        x_new, y_new = brightness(x_new, y_new)

    if np.random.randint(0, 10) < 3:
        x_new, y_new = elastic(x_new, y_new, (10.0, 13.0))
    return x_new, y_new


def binary_combine_aug(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
    """
    Combines the elastic, brightness and rotation deformation augmentations with a 30% chance of each augmentation being
    applied.

    :param x:
    :param y:
    """
    x_new, y_new = x, y

    if np.random.randint(0, 10) < 3:
        x_new, y_new = elastic(x_new, y_new, (2.0, 4.0))

    if np.random.randint(0, 10) < 3:
        x_new, y_new = brightness(x_new, y_new)

    if np.random.randint(0, 10) < 3:
        x_new, y_new = rotation(x_new, y_new)

    return x_new, y_new
