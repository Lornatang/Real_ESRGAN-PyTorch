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
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address
        image_size (int): High resolution image size
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training dataset is for data enhancement,
            and the verification data set is not for data enhancement.
        degradation_model_parameters_dict (dict): Parameter dictionary with degenerate model

    """

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str,
                 degradation_model_parameters_dict: dict) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        # Define degradation model parameters
        self.parameters = degradation_model_parameters_dict
        # Define the size of the sinc filter kernel
        self.sinc_tensor = torch.zeros([self.parameters["sinc_kernel_size"],
                                        self.parameters["sinc_kernel_size"]]).float()
        self.sinc_tensor[self.parameters["sinc_kernel_size"] // 2, self.parameters["sinc_kernel_size"] // 2] = 1
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor
        # Load training dataset or test dataset
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if self.mode == "Train":
            # Image data augmentation
            hr_image = imgproc.random_rotate(image, angles=[0, 90, 180, 270])
            hr_image = imgproc.random_horizontally_flip(hr_image, p=0.5)
            hr_image = imgproc.random_vertically_flip(hr_image, p=0.5)

            # BGR convert to RGB
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            hr_tensor = imgproc.image_to_tensor(hr_image, range_norm=False, half=False)

            # First degenerate operation
            kernel_size1 = random.choice(self.parameters["gaussian_kernel_range"])
            if np.random.uniform() < self.parameters["sinc_kernel_probability1"]:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size1 < int(np.median(self.parameters["gaussian_kernel_range"])):
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel1 = imgproc.sinc_kernel(omega_c, kernel_size1, pad_to=False)
            else:
                kernel1 = imgproc.random_mixed_kernels(
                    self.parameters["gaussian_kernel_type"],
                    self.parameters["gaussian_kernel_probability1"],
                    kernel_size1,
                    self.parameters["gaussian_sigma_range1"],
                    self.parameters["gaussian_sigma_range1"],
                    [-math.pi, math.pi],
                    self.parameters["generalized_kernel_beta_range1"],
                    self.parameters["plateau_kernel_beta_range1"],
                    noise_range=None)
            # pad kernel
            pad_size = (self.parameters["gaussian_kernel_range"][-1] - kernel_size1) // 2
            kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

            # Second degenerate operation
            kernel_size2 = random.choice(self.parameters["gaussian_kernel_range"])
            if np.random.uniform() < self.parameters["sinc_kernel_probability2"]:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size2 < int(np.median(self.parameters["gaussian_kernel_range"])):
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = imgproc.sinc_kernel(omega_c, kernel_size2, pad_to=False)
            else:
                kernel2 = imgproc.random_mixed_kernels(
                    self.parameters["gaussian_kernel_type"],
                    self.parameters["gaussian_kernel_probability2"],
                    kernel_size2,
                    self.parameters["gaussian_sigma_range2"],
                    self.parameters["gaussian_sigma_range2"],
                    [-math.pi, math.pi],
                    self.parameters["generalized_kernel_beta_range2"],
                    self.parameters["plateau_kernel_beta_range2"],
                    noise_range=None)

            # pad kernel
            pad_size = (self.parameters["gaussian_kernel_range"][-1] - kernel_size2) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # Final sinc kernel
            if np.random.uniform() < self.parameters["sinc_kernel_probability3"]:
                kernel_size2 = random.choice(self.parameters["gaussian_kernel_range"])
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = imgproc.sinc_kernel(omega_c, kernel_size2, pad_to=self.parameters["sinc_kernel_size"])
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.sinc_tensor

            kernel1 = torch.FloatTensor(kernel1)
            kernel2 = torch.FloatTensor(kernel2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)

            return {"hr": hr_tensor, "kernel1": kernel1, "kernel2": kernel2, "sinc_kernel": sinc_kernel}

        elif self.mode == "Valid":
            # Center crop image
            hr_image = imgproc.center_crop(image, self.image_size)
            # Use Bicubic kernel create LR image
            lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)

            # BGR convert to RGB
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            lr_tensor = imgproc.image_to_tensor(lr_image, range_norm=False, half=False)
            hr_tensor = imgproc.image_to_tensor(hr_image, range_norm=False, half=False)

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
        lr_tensor = imgproc.image_to_tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image_to_tensor(hr_image, range_norm=False, half=False)

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
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
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
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
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
