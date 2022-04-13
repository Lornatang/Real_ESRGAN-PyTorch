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
import math
import os
import queue
import random
import threading

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int,
                 gaussian_kernel_range: tuple,
                 sinc_kernel_probability1: float, gaussian_kernel_dict1: dict, gaussian_sigma_range1: tuple,
                 generalized_kernel_beta_range1: tuple, plateau_kernel_beta_range1: tuple,
                 sinc_kernel_probability2: float, gaussian_kernel_dict2: dict, gaussian_sigma_range2: tuple,
                 generalized_kernel_beta_range2: tuple, plateau_kernel_beta_range2: tuple,
                 sinc_kernel_probability3: float,
                 resized_probability1: tuple, resized_range1: tuple, gray_noise_probability1: float, gaussian_noise_probability1: float,
                 noise_range1: tuple, poisson_scale_range1: tuple, jpeg_range1: tuple,
                 second_blur_probability: float,
                 resized_probability2: tuple, resized_range2: tuple, gray_noise_probability2: float, gaussian_noise_probability2: float,
                 noise_range2: tuple, poisson_scale_range2: tuple, jpeg_range2: tuple,
                 dataset_mode: str, device: torch.device) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.jpeg_operation = imgproc.DiffJPEG(differentiable=False).to(device)  # simulate JPEG compression artifacts
        self.usm_sharpener = imgproc.USMSharp().to(device)  # do usm sharpening
        # Get all image file names in folder
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Parameter settings during the first degradation operation
        self.gaussian_kernel_range = gaussian_kernel_range
        self.sinc_kernel_probability1 = sinc_kernel_probability1
        self.gaussian_sigma_range1 = gaussian_sigma_range1
        self.gaussian_kernel_dict1 = gaussian_kernel_dict1
        self.generalized_kernel_beta_range1 = generalized_kernel_beta_range1
        self.plateau_kernel_beta_range1 = plateau_kernel_beta_range1
        # Parameter settings during the second degradation operation
        self.sinc_kernel_probability2 = sinc_kernel_probability2
        self.gaussian_sigma_range2 = gaussian_sigma_range2
        self.gaussian_kernel_dict2 = gaussian_kernel_dict2
        self.generalized_kernel_beta_range2 = generalized_kernel_beta_range2
        self.plateau_kernel_beta_range2 = plateau_kernel_beta_range2
        self.sinc_kernel_probability3 = sinc_kernel_probability3
        self.sinc_tensor = torch.zeros(21, 21).float()
        self.sinc_tensor[10, 10] = 1
        # Other degradation mode parameter
        self.resized_probability1 = resized_probability1
        self.resized_range1 = resized_range1
        self.gray_noise_probability1 = gray_noise_probability1
        self.gaussian_noise_probability1 = gaussian_noise_probability1
        self.noise_range1 = noise_range1
        self.poisson_scale_range1 = poisson_scale_range1
        self.jpeg_range1 = jpeg_range1
        self.second_blur_probability = second_blur_probability
        self.resized_probability2 = resized_probability2
        self.resized_range2 = resized_range2
        self.gray_noise_probability2 = gray_noise_probability2
        self.gaussian_noise_probability2 = gaussian_noise_probability2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2
        self.dataset_mode = dataset_mode

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED)

        # Image processing operations
        if self.dataset_mode == "Train":
            hr_image = imgproc.random_crop(image, self.image_size)
            hr_image = imgproc.random_rotate(hr_image, angles=[0, 90, 180, 270])
            hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
            hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)
        elif self.dataset_mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        # First degenerate operation
        kernel_size1 = random.choice(self.gaussian_kernel_range)
        if np.random.uniform() < self.sinc_kernel_probability1:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size1 < int((self.gaussian_kernel_range[0] + self.gaussian_kernel_range[-1]) // 2):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = imgproc.sinc_kernel(omega_c, kernel_size1, pad_to=False)
        else:
            kernel = imgproc.random_mixed_kernels(
                list(self.gaussian_kernel_dict1.keys()),
                list(self.gaussian_kernel_dict1.values()),
                kernel_size1,
                self.gaussian_sigma_range1,
                self.gaussian_sigma_range1,
                [-math.pi, math.pi],
                self.generalized_kernel_beta_range1,
                self.plateau_kernel_beta_range1,
                noise_range=None)
        # pad kernel
        pad_size = (self.gaussian_kernel_range[-1] - kernel_size1) // 2
        kernel1 = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # Second degenerate operation
        kernel_size2 = random.choice(self.gaussian_kernel_range)
        if np.random.uniform() < self.sinc_kernel_probability2:
            if kernel_size2 < int((self.gaussian_kernel_range[0] + self.gaussian_kernel_range[-1]) // 2):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = imgproc.sinc_kernel(omega_c, kernel_size2, pad_to=False)
        else:
            kernel2 = imgproc.random_mixed_kernels(
                list(self.gaussian_kernel_dict2.keys()),
                list(self.gaussian_kernel_dict2.values()),
                kernel_size2,
                self.gaussian_sigma_range2,
                self.gaussian_sigma_range2,
                [-math.pi, math.pi],
                self.generalized_kernel_beta_range2,
                self.plateau_kernel_beta_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.gaussian_kernel_range[-1] - kernel_size2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # Final sinc kernel
        if np.random.uniform() < self.sinc_kernel_probability3:
            kernel_size = random.choice(self.gaussian_kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = imgproc.sinc_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.sinc_tensor

        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)

        # Feed data
        usm_hr_image = self.usm_sharpener(hr_image)

        # First degradation process
        # Gaussian blur
        out = imgproc.blur(usm_hr_image, kernel1)

        # Resize
        updown_type = random.choices(["up", "down", "keep"], self.resized_probability1)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resized_range1[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resized_range1[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Noise
        if np.random.uniform() < self.gaussian_noise_probability1:
            out = imgproc.random_add_gaussian_noise_pt(out, sigma_range=self.noise_range1, clip=True, rounds=False,
                                                       gray_prob=self.gray_noise_probability1)
        else:
            out = imgproc.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range1,
                gray_prob=self.gray_noise_probability1,
                clip=True,
                rounds=False)

        # JPEG
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range1)
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeg_operation(out, quality=jpeg_p)

        # Second degradation process
        # Gaussian blur
        if np.random.uniform() < self.second_blur_probability:
            out = imgproc.blur(out, kernel2)

        # Resize
        updown_type = random.choices(["up", "down", "keep"], self.resized_probability2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resized_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resized_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(int(self.image_size / self.upscale_factor * scale), int(self.image_size / self.upscale_factor * scale)),
                            mode=mode)

        # Noise
        if np.random.uniform() < self.gaussian_noise_probability2:
            out = imgproc.random_add_gaussian_noise_pt(out, sigma_range=self.noise_range2, clip=True, rounds=False,
                                                       gray_prob=self.gray_noise_probability2)
        else:
            out = imgproc.random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_probability2,
                clip=True,
                rounds=False)

        if np.random.uniform() < 0.5:
            # Resize
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(self.image_size // self.upscale_factor, self.image_size // self.upscale_factor), mode=mode)
            # Final sinc blur
            out = imgproc.blur(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeg_operation(out, quality=jpeg_p)
        else:
            # JPEG
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeg_operation(out, quality=jpeg_p)
            # Resize
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(self.image_size // self.upscale_factor, self.image_size // self.upscale_factor), mode=mode)
            # Final sinc blur
            out = imgproc.blur(out, sinc_kernel)

        # clamp and round
        lr_image = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(usm_hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
    """

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.lr_image_file_names = [os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)]
        self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_lr_image_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        lr_image = cv2.imread(self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        hr_image = cv2.imread(self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # BGR convert to RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
