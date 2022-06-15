# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import itertools
import math
import random
from typing import Any

import cv2
import numpy as np
import torch
from scipy import special
from scipy.stats import multivariate_normal
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from torchvision.transforms.functional_tensor import rgb_to_grayscale

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "image_resize",
    "expand_y", "rgb2ycbcr", "bgr2ycbcr", "ycbcr2bgr", "ycbcr2rgb",
    "rgb2ycbcr_torch", "bgr2ycbcr_torch",
    "center_crop", "random_crop", "random_rotate", "random_horizontally_flip", "random_vertically_flip",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def _calculate_rotate_sigma_matrix(sigma_x: float, sigma_y: float, theta: float):
    """Calculate rotated sigma matrix.

    Args:
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement

    Returns:
        np.ndarray: Rotated sigma matrix

    """
    d_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

    return out


def _mesh_grid(kernel_size: int):
    """Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Args:
        kernel_size (int): Gaussian kernel size

    Returns:
        xy (np.ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (np.ndarray): with the shape (kernel_size, kernel_size)
        yy (np.ndarray): with the shape (kernel_size, kernel_size)

    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(
        kernel_size, kernel_size, 2)

    return xy, xx, yy


def _calculate_probability_density_function(sigma_matrix: np.ndarray, grid: np.ndarray):
    """Calculate probability density function of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (np.ndarray): with the shape (2, 2)
        grid (np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        kernel (np.ndarray): un-normalized kernel

    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    probability_density_function = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

    return probability_density_function


def _calculate_cumulative_density_function(skew_matrix: np.ndarray, grid: np.ndarray):
    """Calculate the CDF of the standard bivariate Gaussian distribution.
        Used in skewed Gaussian distribution.

    Args:
        skew_matrix (np.ndarray): skew matrix
        grid (ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        cumulative_density_function (ndarray): Cumulative density function

    """
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, skew_matrix)
    cumulative_density_function = rv.cdf(grid)

    return cumulative_density_function


def _generate_bivariate_gaussian_kernel(kernel_size, sigma_x: float, sigma_y: float, theta: float,
                                        grid: np.ndarray = None, isotropic: bool = True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sigma_x` is used. `sigma_y` and `theta` is ignored.

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement
        grid (np.ndarray, optional): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        gaussian_kernel (np.ndarray): Gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    bivariate_gaussian_kernel = _calculate_probability_density_function(sigma_matrix, grid)
    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


def _generate_bivariate_generalized_gaussian_kernel(kernel_size: int, sigma_x: float, sigma_y: float, theta: float,
                                                    beta: float,
                                                    grid: np.ndarray = None, isotropic: bool = True):
    """Generate a bivariate generalized Gaussian kernel.
    Described in `Parameter Estimation For Multivariate Generalized Gaussian Distributions`_ by Pascal et. al (2013).
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution
        grid (ndarray, optional): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): bivariate generalized gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_generalized_gaussian_kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel)

    return bivariate_generalized_gaussian_kernel


# Implementation reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py`
def _generate_bivariate_plateau_gaussian_kernel(kernel_size: int, sigma_x: float, sigma_y: float, theta: float,
                                                beta: float,
                                                grid: np.ndarray = None, isotropic: bool = True):
    """Generate a plateau-like anisotropic kernel.
    In the isotropic mode, only `sigma_x` is used. `sigma_y` and `theta` is ignored.

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (np.ndarray, optional): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (np.ndarray): Bivariate plateau gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_plateau_gaussian_kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


# Implementation reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py`
def random_bivariate_gaussian_kernel(kernel_size: int,
                                     sigma_x_range: tuple,
                                     sigma_y_range: tuple,
                                     rotation_range: tuple,
                                     noise_range=None,
                                     isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        noise_range(tuple, optional): multiplicative kernel noise. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_gaussian_kernel (np.ndarray): Bivariate gaussian kernel

    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    bivariate_gaussian_kernel = _generate_bivariate_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation,
                                                                    isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_gaussian_kernel.shape)
        bivariate_gaussian_kernel = bivariate_gaussian_kernel * noise

    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


# Implementation reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py`
def random_bivariate_generalized_gaussian_kernel(kernel_size: int,
                                                 sigma_x_range: tuple,
                                                 sigma_y_range: tuple,
                                                 rotation_range: tuple,
                                                 beta_range: tuple,
                                                 noise_range=None,
                                                 isotropic=True):
    """Randomly generate bivariate generalized Gaussian kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        beta_range (tuple):
        noise_range(tuple, optional): multiplicative kernel noise. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): Bivariate generalized gaussian kernel
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_generalized_gaussian_kernel = _generate_bivariate_generalized_gaussian_kernel(kernel_size, sigma_x,
                                                                                            sigma_y, rotation, beta,
                                                                                            isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_generalized_gaussian_kernel.shape)
        bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel * noise

    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel)

    return bivariate_generalized_gaussian_kernel


# Implementation reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py`
def random_bivariate_plateau_gaussian_kernel(kernel_size: int,
                                             sigma_x_range: tuple,
                                             sigma_y_range: tuple,
                                             rotation_range: tuple,
                                             beta_range: tuple,
                                             noise_range=None,
                                             isotropic=True):
    """Randomly generate bivariate plateau kernels.
    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        beta_range (tuple):
        noise_range(tuple, optional): multiplicative kernel noise. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (ndarray): Bivariate plateau gaussian kernel

    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_plateau_gaussian_kernel = _generate_bivariate_plateau_gaussian_kernel(kernel_size, sigma_x, sigma_y,
                                                                                    rotation, beta, isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_plateau_gaussian_kernel.shape)
        bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel * noise

    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


# Implementation reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py`
def random_mixed_kernels(kernel_type: list,
                         kernel_prob: float,
                         kernel_size: int,
                         sigma_x_range: list,
                         sigma_y_range: list,
                         rotation_range: list,
                         generalized_kernel_beta_range: list,
                         plateau_kernel_beta_range: list,
                         noise_range: None):
    """Randomly generate mixed kernels

    Args:
        kernel_type (tuple): a list name of gaussian kernel types
        kernel_prob (tuple): corresponding kernel probability for each kernel type
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis. Default:
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        generalized_kernel_beta_range (tuple):
        plateau_kernel_beta_range (tuple):
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None

    Returns:
        mixed kernels (np.ndarray): Mixed kernels

    """
    kernel_type = random.choices(kernel_type, kernel_prob)[0]
    if kernel_type == "isotropic":
        mixed_kernels = random_bivariate_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range, rotation_range,
                                                         noise_range=noise_range,
                                                         isotropic=True)
    elif kernel_type == "anisotropic":
        mixed_kernels = random_bivariate_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range, rotation_range,
                                                         noise_range=noise_range,
                                                         isotropic=False)
    elif kernel_type == "generalized_isotropic":
        mixed_kernels = random_bivariate_generalized_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range,
                                                                     rotation_range,
                                                                     generalized_kernel_beta_range,
                                                                     noise_range=noise_range, isotropic=True)
    elif kernel_type == "generalized_anisotropic":
        mixed_kernels = random_bivariate_generalized_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range,
                                                                     rotation_range,
                                                                     generalized_kernel_beta_range,
                                                                     noise_range=noise_range, isotropic=False)
    elif kernel_type == "plateau_isotropic":
        mixed_kernels = random_bivariate_plateau_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range,
                                                                 rotation_range, plateau_kernel_beta_range,
                                                                 noise_range=None, isotropic=True)
    elif kernel_type == "plateau_anisotropic":
        mixed_kernels = random_bivariate_plateau_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range,
                                                                 rotation_range, plateau_kernel_beta_range,
                                                                 noise_range=None, isotropic=False)
    else:
        mixed_kernels = random_bivariate_gaussian_kernel(kernel_size, sigma_x_range, sigma_y_range, rotation_range,
                                                         noise_range=noise_range,
                                                         isotropic=True)

    return mixed_kernels


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def sinc_kernel(cutoff: float, kernel_size: int, pad_to: int = 0):
    """2D sinc filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd
        pad_to (int): pad kernel size to desired size, must be odd or zero. Default: 0

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    np.seterr(divide="ignore", invalid="ignore")
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)) / (2 * np.pi * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def generate_gaussian_noise(image, sigma=10, gray_noise=False):
    """Generate Gaussian noise

    Args:
        image (np.array): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.
        gray_noise (bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        gaussian_noise (np.array): Gaussian_noise

    """
    if gray_noise:
        gaussian_noise = np.float32(np.random.randn(*(image.shape[0:2]))) * sigma / 255.
        gaussian_noise = np.expand_dims(gaussian_noise, axis=2).repeat(3, axis=2)
    else:
        gaussian_noise = np.float32(np.random.randn(*image.shape)) * sigma / 255.

    return gaussian_noise


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def add_gaussian_noise(image: np.ndarray, sigma=10, clip=True, rounds=False, gray_noise=False):
    """Add Gaussian noise

    Args:
        image (np.ndarray): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.
        clip (bool): Whether to clip image.If `True`, clip image pixel to [0, 1] or [0, 255]. Default: True
        rounds (bool):Default: False
        gray_noise (bool):Default: False

    Returns:
        gaussian_noise (np.array): Gaussian noise

    """
    noise = generate_gaussian_noise(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def generate_gaussian_noise_pt(image: torch.Tensor, sigma: float = 10, gray_noise=0):
    """Add Gaussian noise (PyTorch version)

    Args:
        image (torch.Tensor): Shape (b, c, h, w), range[0, 1], float32.
        sigma (float): Default: 10.
        gray_noise (float): Default: 0

    Returns:
        gaussian_noise (torch.Tensor): Gaussian noise

    """
    b, _, h, w = image.size()

    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(image.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*image.size()[2:4], dtype=image.dtype, device=image.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(*image.size(), dtype=image.dtype, device=image.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise

    return noise


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def add_gaussian_noise_pt(image, sigma=10, gray_noise=0, clip=True, rounds=False):
    """Add Gaussian noise (PyTorch version)

    Args:
        image (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    noise = generate_gaussian_noise_pt(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def random_generate_gaussian_noise(image, sigma_range=(0, 10), gray_prob=0):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    gaussian_noise = generate_gaussian_noise(image, sigma, gray_noise)

    return gaussian_noise


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def random_add_gaussian_noise(image, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise(image, sigma_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def random_generate_gaussian_noise_pt(image, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(image.size(0), dtype=image.dtype, device=image.device) * (sigma_range[1] - sigma_range[0]) + \
            sigma_range[0]
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    gaussian_noise = generate_gaussian_noise_pt(image, sigma, gray_noise)

    return gaussian_noise


# Implementation reference `https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter`
def random_add_gaussian_noise_pt(image, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(image, sigma_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def generate_poisson_noise(image, scale=1.0, gray_noise=False):
    """Generate poisson noise

    Args:
        image (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.

    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32

    """
    if gray_noise:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # round and clip image for counting vals correctly
    image = np.clip((image * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(image * vals) / float(vals))
    noise = out - image

    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)

    poisson_noise = noise * scale

    return poisson_noise


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def add_poisson_noise(image, scale=1.0, clip=True, rounds=False, gray_noise=False):
    """Add poisson noise.

    Args:
        image (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.

    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.

    """
    noise = generate_poisson_noise(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def generate_poisson_noise_pt(image, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)
    Args:
        image (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = image.size()

    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(image, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    image = torch.clamp((image * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(image[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = image.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(image * vals) / vals
    noise = out - image

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)

    poisson_noise = noise * scale

    return poisson_noise


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def add_poisson_noise_pt(image, scale=1.0, clip=True, rounds=False, gray_noise=0):
    """Add poisson noise to a batch of images (PyTorch version).
    Args:
        image (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.
    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise_pt(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def random_generate_poisson_noise(image, scale_range=(0, 1.0), gray_prob=0):
    scale = np.random.uniform(scale_range[0], scale_range[1])

    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    poisson_noise = generate_poisson_noise(image, scale, gray_noise)

    return poisson_noise


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def random_add_poisson_noise(image, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise(image, scale_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def random_generate_poisson_noise_pt(image, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(image.size(0), dtype=image.dtype, device=image.device) * (scale_range[1] - scale_range[0]) + \
            scale_range[0]
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    poisson_noise = generate_poisson_noise_pt(image, scale, gray_noise)

    return poisson_noise


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def random_add_poisson_noise_pt(image, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(image, scale_range, gray_prob)
    out = image + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def add_jpg_compression(image, quality=90):
    """Add JPG compression artifacts.
    Args:
        image (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    image = np.clip(image, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image * 255., encode_param)
    image = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return image


# Implementation reference `https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219`
def random_add_jpg_compression(image, quality_range=(90, 100)):
    """Randomly add JPG compression artifacts.
    Args:
        image (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(image, quality)


# ------------------------ utils ------------------------#
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = nn.Parameter(torch.from_numpy(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Args:
        quality(float): Quality for jpeg compression.
    Returns:
        float: Compression factor.
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# ------------------------ compression ------------------------#
class RGB2YCbCrJpeg(nn.Module):
    """ Converts RGB image to YCbCr
    """

    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(Tensor): batch x 3 x height x width
        Returns:
            Tensor: batch x height x width x 3
        """
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class ChromaSubsampling(nn.Module):
    """ Chroma subsampling on CbCr channels
    """

    def __init__(self):
        super(ChromaSubsampling, self).__init__()

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3
        Returns:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image_2[:, 1, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cr = F.avg_pool2d(image_2[:, 2, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Module):
    """ Splitting image into patches
    """

    def __init__(self):
        super(BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class DCT8x8(nn.Module):
    """ Discrete Cosine Transformation
    """

    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class YQuantize(nn.Module):
    """ JPEG Quantization for Y channel
    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CQuantize(nn.Module):
    """ JPEG Quantization for CbCr channels
    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CompressJpeg(nn.Module):
    """Full JPEG compression algorithm
    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x 3 x height x width
        Returns:
            dict(tensor): Compressed tensor with batch x h*w/64 x 8 x 8.
        """
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# ------------------------ decompression ------------------------#


class YDequantize(nn.Module):
    """Dequantize Y channel
    """

    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = y_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class CDequantize(nn.Module):
    """Dequantize CbCr channel
    """

    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = c_table

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            out = image * (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class iDCT8x8(nn.Module):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self):
        super(iDCT8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width
        Returns:
            Tensor: batch x height x width
        """
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class BlockMerging(nn.Module):
    """Merge patches into image
    """

    def __init__(self):
        super(BlockMerging, self).__init__()

    def forward(self, patches, height, width):
        """
        Args:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)
        Returns:
            Tensor: batch x height x width
        """
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
    """Upsample chroma layers
    """

    def __init__(self):
        super(ChromaUpsampling, self).__init__()

    def forward(self, y, cb, cr):
        """
        Args:
            y(tensor): y channel image
            cb(tensor): cb channel
            cr(tensor): cr channel
        Returns:
            Tensor: batch x height x width x 3
        """

        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCr2RGBJpeg(nn.Module):
    """Converts YCbCr image to RGB JPEG
    """

    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()

        matrix = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3
        Returns:
            Tensor: batch x 3 x height x width
        """
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class DeCompressJpeg(nn.Module):
    """Full JPEG decompression algorithm
    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        """
        Args:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            imgh(int)
            imgw(int)
            factor(float)
        Returns:
            Tensor: batch x 3 x height x width
        """
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255


class DiffJPEG(nn.Module):
    """This JPEG algorithm result is slightly different from cv2.
    DiffJPEG supports batch processing.
    Args:
        differentiable(bool): If True, uses custom differentiable rounding function, if False, uses standard torch.round
    """

    def __init__(self, differentiable=True):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round

        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def forward(self, x, quality):
        """
        Args:
            x (Tensor): Input image, bchw, rgb, [0, 1]
            quality(float): Quality factor for jpeg compression scheme.
        """
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            for i in range(factor.size(0)):
                factor[i] = quality_to_factor(factor[i])
        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0
        # why should use 16
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
        recovered = recovered[:, :, 0:h, 0:w]
        return recovered


def blur(image, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        image (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = image.size()
    if k % 2 == 1:
        image = F.pad(image, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = image.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        image = image.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(image, kernel, padding=0).view(b, c, h, w)
    else:
        image = image.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(image, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.
    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        usm_blur = blur(img, self.kernel)
        residual = img - usm_blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = blur(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _cubic(x: Any):
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation.
    """

    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _calculate_weights_indices(in_length: int, out_length: int, scale: float, kernel_width: int, antialiasing: bool):
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(out_length,
                                                                                                             p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        np.ndarray: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format,
    and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC form

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr2ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def rgb2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[65.481, -37.797, 112.0],
                               [128.553, -74.203, -93.786],
                               [24.966, 112.0, -18.214]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of bgr2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[24.966, 112.0, -18.214],
                               [128.553, -74.203, -93.786],
                               [65.481, -37.797, 112.0]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        np.ndarray: Small patch image.
    """

    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(lr_images: torch.Tensor, hr_images: torch.Tensor,
                hr_image_size: int, upscale_factor: int) -> [torch.Tensor, torch.Tensor]:
    """Crop small image patches from one image.

    Args:
        lr_images (torch.Tensor): Low resolution images
        hr_images (torch.Tensor): High resolution images
        hr_image_size (int): The size of the captured high-resolution image area.
        upscale_factor (int): How many times the high-resolution image should be the low-resolution image

    Returns:
        patch_lr_images, patch_hr_images (torch.Tensor, torch.Tensor): Small lr patch images, Small hr patch images

    """
    hr_image_height, hr_image_width = hr_images[0].size()[1:]

    # Just need to find the top and left coordinates of the image
    hr_top = random.randint(0, hr_image_height - hr_image_size)
    hr_left = random.randint(0, hr_image_width - hr_image_size)

    # Define the LR image position
    lr_top = hr_top // upscale_factor
    lr_left = hr_left // upscale_factor
    lr_image_size = hr_image_size // upscale_factor

    # Create patch images
    patch_lr_images = torch.zeros([lr_images.shape[0], lr_images.shape[1], lr_image_size, lr_image_size],
                                  dtype=lr_images.dtype,
                                  device=lr_images.device)
    patch_hr_images = torch.zeros([hr_images.shape[0], hr_images.shape[1], hr_image_size, hr_image_size],
                                  dtype=lr_images.dtype,
                                  device=hr_images.device)

    # Crop image patch
    for i in range(lr_images.shape[0]):
        patch_lr_images[i, :, :, :] = lr_images[i, :, lr_top:lr_top + lr_image_size, lr_left:lr_left + lr_image_size]
        patch_hr_images[i, :, :, :] = hr_images[i, :, hr_top:hr_top + hr_image_size, hr_left:hr_left + hr_image_size]

    return patch_lr_images, patch_hr_images


def random_rotate(image,
                  angles: list,
                  center: tuple[int, int] = None,
                  scale_factor: float = 1.0) -> np.ndarray:
    """Rotate an image by a random angle

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image
