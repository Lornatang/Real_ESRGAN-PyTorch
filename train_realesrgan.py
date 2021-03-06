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
from torch.nn import functional as F, BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import imgproc
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import NIQE
from model import Generator, Discriminator, EMA, ContentLoss


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load dataset successfully.")

    discriminator, generator, ema_model = build_model()
    print("Build all model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    if config.resume:
        print("Loading RealESRNet model weights")
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        generator.load_state_dict(checkpoint["state_dict"])
        print("Loaded RealESRNet model weights.")

    print("Check whether the pretrained discriminator model is restored...")
    if config.resume_d:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_d, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Load the optimizer model
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained discriminator model weights.")

    print("Check whether the pretrained generator model is restored...")
    if config.resume_g:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_g, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        generator.load_state_dict(model_state_dict)
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained generator model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        train(discriminator,
              generator,
              ema_model,
              train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer)
        _ = validate(generator, ema_model, valid_prefetcher, epoch, writer, niqe_model, "Valid")
        niqe = validate(generator, ema_model, test_prefetcher, epoch, writer, niqe_model, "Test")
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save({"epoch": epoch + 1,
                    "best_niqe": best_niqe,
                    "state_dict": discriminator.state_dict(),
                    "optimizer": d_optimizer.state_dict(),
                    "scheduler": d_scheduler.state_dict()},
                   os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"))
        torch.save({"epoch": epoch + 1,
                    "best_niqe": best_niqe,
                    "state_dict": generator.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer": g_optimizer.state_dict(),
                    "scheduler": g_scheduler.state_dict()},
                   os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "d_best.pth.tar"))
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "d_last.pth.tar"))
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
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

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


def build_model() -> [nn.Module, nn.Module, nn.Module]:
    discriminator = Discriminator()
    generator = Generator(config.in_channels, config.out_channels, config.upscale_factor)

    # Transfer to CUDA
    discriminator = discriminator.to(device=config.device)
    generator = generator.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_model = EMA(generator, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device)
    ema_model.register()

    return discriminator, generator, ema_model


def define_loss() -> [nn.L1Loss, ContentLoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = ContentLoss(config.feature_model_extractor_nodes,
                                    config.feature_model_normalize_mean,
                                    config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=config.device)
    content_criterion = content_criterion.to(device=config.device)
    adversarial_criterion = adversarial_criterion.to(device=config.device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(discriminator.parameters(), config.model_lr, config.model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.model_lr, config.model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.MultiStepLR,
                                                                           lr_scheduler.MultiStepLR]:
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma)

    return d_scheduler, g_scheduler


def train(discriminator: nn.Module,
          generator: nn.Module,
          ema_model: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          pixel_criterion: nn.L1Loss,
          content_criterion: ContentLoss,
          adversarial_criterion: BCEWithLogitsLoss,
          d_optimizer: optim.Adam,
          g_optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        discriminator (nn.Module): discriminator model in adversarial networks
        generator (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        content_criterion (ContentLoss): Calculate the feature difference between real samples and fake samples by the feature extraction model
        adversarial_criterion (nn.BCEWithLogitsLoss): Calculate the semantic difference between real samples and fake samples by the discriminator model
        d_optimizer (optim.Adam): an optimizer for optimizing discriminator models in adversarial networks
        g_optimizer (optim.Adam): an optimizer for optimizing generator models in adversarial networks
        epoch (int): number of training epochs during training the adversarial network
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
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put all model in train mode.
    discriminator.train()
    generator.train()

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

        # Sharpen high-resolution images
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
        out = jpeg_operation(out, quality=quality)

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
            out = jpeg_operation(out, quality=quality)
        else:
            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(*config.degradation_process_parameters_dict["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)

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

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = hr.shape
        real_label = torch.full([batch_size, 1, height, width], 1.0, dtype=torch.float, device=config.device)
        fake_label = torch.full([batch_size, 1, height, width], 0.0, dtype=torch.float, device=config.device)

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = generator(lr)
            pixel_loss = config.pixel_weight * pixel_criterion(usm_sharpener(sr, 0.5, 10), hr)
            content_loss = torch.sum(torch.mul(torch.Tensor(config.content_weight),
                                               torch.Tensor(content_criterion(usm_sharpener(sr, 0.5, 10), hr))))
            adversarial_loss = config.adversarial_weight * adversarial_criterion(discriminator(sr), real_label)
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()
        # Finish training the generator model

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(d_loss_hr).backward()

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            # Calculate the total discriminator loss value
            d_loss = d_loss_sr + d_loss_hr
        # Call the gradient scaling function in the mixed precision API to
        # bp the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()
        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Update EMA
        ema_model.update()

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
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
