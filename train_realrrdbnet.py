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
from model import Generator, EMA


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    model = build_model()
    print("Build RRDBNet model successfully.")

    psnr_criterion, pixel_criterion = define_loss()
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
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
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

    # Create an Exponential Moving Average Model
    ema_model = EMA(model, config.model_weight_decay)
    ema_model.register()

    for epoch in range(config.start_epoch, config.epochs):
        train(model, ema_model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, ema_model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, ema_model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "state_dict": model.state_dict(),
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


def build_model() -> nn.Module:
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor).to(config.device)

    return model


def define_loss() -> [nn.MSELoss, nn.L1Loss]:
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.L1Loss().to(config.device)

    return psnr_criterion, pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)

    return scheduler


def train(model, ema_model,
          train_prefetcher, psnr_criterion, pixel_criterion,
          optimizer, epoch, scaler, writer) -> None:
    # Defining JPEG image manipulation methods
    jpeg_operation = imgproc.DiffJPEG(differentiable=False).to(config.device, non_blocking=True)
    # Define image sharpening method
    usm_sharpener = imgproc.USMSharp().to(config.device, non_blocking=True)
    # Calculate how many iterations there are under epoch
    batches = len(train_prefetcher)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    # enable preload
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    while batch_data is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        hr = batch_data["hr"].to(config.device, non_blocking=True)
        kernel1 = batch_data["kernel1"].to(config.device, non_blocking=True)
        kernel2 = batch_data["kernel2"].to(config.device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(config.device, non_blocking=True)

        # Feed data
        out = usm_sharpener(hr)

        # Get original image size
        image_height, image_width = out.size()[2:4]

        # First degradation process
        # Gaussian blur
        if np.random.uniform() <= config.degradation_process_parameters_dict["first_blur_probability"]:
            out = imgproc.blur(out, kernel1)

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
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range1"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability1"])
        else:
            out = imgproc.random_add_poisson_noise_pt(
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
            out = imgproc.blur(out, kernel2)

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
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict["gray_noise_probability2"])
        else:
            out = imgproc.random_add_poisson_noise_pt(
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
            out = imgproc.blur(out, sinc_kernel)

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
            out = imgproc.blur(out, sinc_kernel)

        # Clamp and round
        lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # LR and HR crop the specified area respectively
        lr, hr = imgproc.random_crop(lr, hr, config.image_size, config.upscale_factor)

        # Initialize the generator gradient
        model.zero_grad()

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Gradient zoom
        scaler.scale(loss).backward()
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update()

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

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


def validate(model, ema_model, valid_prefetcher, psnr_criterion, epoch, writer, mode) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_prefetcher), [batch_time, psnres], prefix=f"{mode}: ")

    # Restore the model before the EMA
    ema_model.apply_shadow()
    # Put the model in verification mode
    model.eval()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    with torch.no_grad():
        # enable preload
        valid_prefetcher.reset()
        batch_data = valid_prefetcher.next()

        while batch_data is not None:
            # measure data loading time
            lr = batch_data["lr"].to(config.device, non_blocking=True)
            hr = batch_data["hr"].to(config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Convert RGB tensor to Y tensor
            sr_image = imgproc.tensor2image(sr, range_norm=False, half=True)
            sr_image = sr_image.astype(np.float32) / 255.
            sr_y_image = imgproc.rgb2ycbcr(sr_image, use_y_channel=True)
            sr_y_tensor = imgproc.image2tensor(sr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)

            hr_image = imgproc.tensor2image(hr, range_norm=False, half=True)
            hr_image = hr_image.astype(np.float32) / 255.
            hr_y_image = imgproc.rgb2ycbcr(hr_image, use_y_channel=True)
            hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr_y_tensor, hr_y_tensor))
            psnres.update(psnr.item(), lr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.print_frequency == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = valid_prefetcher.next()

            # After a batch of data is calculated, add 1 to the number of batches
            batch_index += 1

    # Restoring the EMA model
    ema_model.restore()

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg


# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
