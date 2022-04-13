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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_rrdbnet"
# Experiment name, easy to save weights and log files
exp_name = "RRDBNet_baseline"
# Degradation model parameters
gaussian_kernel_range = [7, 9, 11, 13, 15, 17, 19, 21]
sinc_kernel_size = 21

gaussian_kernel_size1 = 21
gaussian_kernel_dict1 = {
    "isotropic": 0.45,
    "anisotropic ": 0.25,
    "generalized_isotropic": 0.12,
    "generalized_anisotropic": 0.03,
    "plateau_isotropic": 0.12,
    "plateau_anisotropic": 0.03,
}
sinc_kernel_probability1 = 0.1
gaussian_sigma_range1 = [0.2, 3]
generalized_kernel_beta_range1 = [0.5, 4]
plateau_kernel_beta_range1 = [1, 2]

gaussian_kernel_size2 = 21
gaussian_kernel_dict2 = {
    "isotropic": 0.45,
    "anisotropic ": 0.25,
    "generalized_isotropic": 0.12,
    "generalized_anisotropic": 0.03,
    "plateau_isotropic": 0.12,
    "plateau_anisotropic": 0.03,
}
sinc_kernel_probability2 = 0.1
min_gaussian_sigma2 = 0.2
max_gaussian_sigma2 = 1.5
min_generalized_kernel_beta2 = 0.5
max_generalized_kernel_beta2 = 4
min_plateau_kernel_beta2 = 1
max_plateau_kernel_beta2 = 2

sinc_kernel_probability3 = 0.8

# First degradation
resize_probability1: [0.2, 0.7, 0.1]  # up, down, keep
resize_range1: [0.15, 1.5]
gray_noise_probability1: 0.4
gaussian_noise_probability1: 0.5
noise_range1: [1, 30]
poisson_scale_range1: [0.05, 3]
jpeg_range1: [30, 95]

# Second degradation
second_blur_probability: 0.8
resize_probability2: [0.3, 0.4, 0.3]  # up, down, keep
resize_range2: [0.3, 1.2]
gaussian_noise_probability2: 0.5
noise_range2: [1, 25]
poisson_scale_range2: [0.05, 2.5]
gray_noise_probability2: 0.4
jpeg_range2: [30, 95]

if mode == "train_rrdbnet":
    # Dataset address
    train_image_dir = "data/DIV2K/RealESRGAN/train"
    valid_image_dir = "data/DIV2K/RealESRGAN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 256
    batch_size = 48
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 44

    # Optimizer parameter
    model_lr = 2e-4
    model_betas = (0.9, 0.99)

    print_frequency = 1000

if mode == "train_realesrgan":
    # Dataset address
    train_image_dir = "data/DIV2K/RealESRGAN/train"
    valid_image_dir = "data/DIV2K/RealESRGAN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 256
    batch_size = 48
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = "results/RRDBNet_baseline/g_last.pth.tar"
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 9

    # Feature extraction layer parameter configuration
    feature_extractor_node = "features.34"
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # Loss function weight
    pixel_weight = 0.01
    feature_weight = 1.0
    adversarial_weight = 0.005

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)

    # LR scheduler parameter
    lr_scheduler_step_size = epochs // 2
    lr_scheduler_gamma = 0.1

    print_frequency = 100

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/g_last.pth.tar"
