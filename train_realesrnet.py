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
import os
import random
import shutil
import time
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import imgproc
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import NIQE
from model import Generator, EMA


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    model, ema_model = build_model()
    print("Build all model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the optimizer scheduler
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train(model, ema_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, ema_model, valid_prefetcher, epoch, writer, niqe_model, "Valid")
        niqe = validate(model, ema_model, test_prefetcher, epoch, writer, niqe_model, "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save({"epoch": epoch + 1,
                    "best_niqe": best_niqe,
                    "state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir,
                                            config.image_size,
                                            config.upscale_factor,
                                            "Train",
                                            config.degradation_model_parameters_dict)
    valid_datasets = TrainValidImageDataset(config.valid_image_dir,
                                            config.image_size,
                                            config.upscale_factor,
                                            "Valid",
                                            config.degradation_model_parameters_dict)
    test_datasets = TestImageDataset(config.test_lr_image_dir,
                                     config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_model = EMA(model, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device)
    ema_model.register()

    return model, ema_model


def define_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=config.device)

    return pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)

    return scheduler


def train(model: nn.Module,
          ema_model: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          pixel_criterion: nn.MSELoss,
          optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        model (nn.Module): the generator model in the generative network
        ema_model (nn.Module): Exponential Moving Average Model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        optimizer (optim.Adam): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Defining JPEG image manipulation methods
    jpeg_operation = imgproc.DiffJPEG(False)
    jpeg_operation = jpeg_operation.to(device=config.device)
    # Define image sharpening method
    usm_sharpener = imgproc.USMSharp(50, 0)
    usm_sharpener = usm_sharpener.to(device=config.device)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        hr = batch_data["hr"].to(device=config.device, non_blocking=True)
        kernel1 = batch_data["kernel1"].to(device=config.device, non_blocking=True)
        kernel2 = batch_data["kernel2"].to(device=config.device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(device=config.device, non_blocking=True)

        # # Sharpen high-resolution images
        out = usm_sharpener(hr, 0.5, 10)

        # Get original image size
        image_height, image_width = out.size()[2:4]

        # First degradation process
        # Gaussian blur
        if np.random.uniform() <= config.degradation_process_parameters_dict["first_blur_probability"]:
            out = imgproc.filter2d_torch(out, kernel1)

        # Resize
        updown_type = random.choices(["up", "down", "keep"],
                                     config.degradation_process_parameters_dict["resize_probability1"])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, config.degradation_process_parameters_dict["resize_range1"][1])
        elif updown_type == "down":
            scale = np.random.uniform(config.degradation_process_parameters_dict["resize_range1"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Noise
        if np.random.uniform() < config.degradation_process_parameters_dict["gaussian_noise_probability1"]:
            out = imgproc.random_add_gaussian_noise_torch(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range1"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability1"])
        else:
            out = imgproc.random_add_poisson_noise_torch(
                image=out,
                scale_range=config.degradation_process_parameters_dict["poisson_scale_range1"],
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability1"],
                clip=True,
                rounds=False)

        # JPEG
        quality = out.new_zeros(out.size(0)).uniform_(*config.degradation_process_parameters_dict["jpeg_range1"])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeg_operation(out, quality)

        # Second degradation process
        # Gaussian blur
        if np.random.uniform() < config.degradation_process_parameters_dict["second_blur_probability"]:
            out = imgproc.filter2d_torch(out, kernel2)

        # Resize
        updown_type = random.choices(["up", "down", "keep"],
                                     config.degradation_process_parameters_dict["resize_probability2"])[0]
        if updown_type == "up":
            scale = np.random.uniform(1, config.degradation_process_parameters_dict["resize_range2"][1])
        elif updown_type == "down":
            scale = np.random.uniform(config.degradation_process_parameters_dict["resize_range2"][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out,
                            size=(int(image_height / config.upscale_factor * scale),
                                  int(image_width / config.upscale_factor * scale)),
                            mode=mode)

        # Noise
        if np.random.uniform() < config.degradation_process_parameters_dict["gaussian_noise_probability2"]:
            out = imgproc.random_add_gaussian_noise_torch(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability2"])
        else:
            out = imgproc.random_add_poisson_noise_torch(
                image=out,
                scale_range=config.degradation_process_parameters_dict["poisson_scale_range2"],
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability2"],
                clip=True,
                rounds=False)

        if np.random.uniform() < 0.5:
            # Resize
            out = F.interpolate(out,
                                size=(image_height // config.upscale_factor, image_width // config.upscale_factor),
                                mode=random.choice(["area", "bilinear", "bicubic"]))
            # Sinc blur
            out = imgproc.filter2d_torch(out, sinc_kernel)

            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(*config.degradation_process_parameters_dict["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality)
        else:
            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(*config.degradation_process_parameters_dict["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality)

            # Resize
            out = F.interpolate(out,
                                size=(image_height // config.upscale_factor, image_width // config.upscale_factor),
                                mode=random.choice(["area", "bilinear", "bicubic"]))

            # Sinc blur
            out = imgproc.filter2d_torch(out, sinc_kernel)

        # Clamp and round
        lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # LR and HR crop the specified area respectively
        lr, hr = imgproc.random_crop(lr, hr, config.image_size, config.upscale_factor)

        # Initialize the generator gradient
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % config.print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After a batch of data is calculated, add 1 to the number of batches
        batch_index += 1


def validate(model: nn.Module,
             ema_model: nn.Module,
             data_prefetcher: CUDAPrefetcher,
             epoch: int,
             writer: SummaryWriter,
             niqe_model: Any,
             mode: str) -> float:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        niqe_model (nn.Module): The model used to calculate the model NIQE metric
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    niqe_metrics = AverageMeter("NIQE", ":4.2f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, niqe_metrics], prefix=f"{mode}: ")

    # Restore the model before the EMA
    ema_model.apply_shadow()
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            niqe = niqe_model(sr)
            niqe_metrics.update(niqe.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # Restoring the EMA model
    ema_model.restore()

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/NIQE", niqe_metrics.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return niqe_metrics.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
