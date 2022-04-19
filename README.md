# Real_ESRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833v2).

## Table of contents

- [Real_ESRGAN-PyTorch](#real_esrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
        - [Train RealESRNet model](#train-realesrnet-model)
        - [Train RealESRGAN model](#train-realesrgan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](#real-esrgan-training-real-world-blind-super-resolution-with-pure-synthetic-data)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the `config.py` file as follows.

- line 77: `upscale_factor` change to the magnification you need to enlarge.
- line 79: `mode` change Set to valid mode.
- line 157: `model_path` change weight address after training.

## Train

Modify the contents of the `config.py` file as follows.

- line 77: `upscale_factor` change to the magnification you need to enlarge.
- line 79: `mode` change set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

### Train RealESRNet model

- line 93: `start_epoch` change number of RealRRDBNet training iterations in the previous round.
- line 94: `resume` change to RealRRDBNet weight address that needs to be loaded.

### Train RealESRGAN model

- line 122: `start_epoch` change number of Real_ESRGAN training iterations in the previous round.
- line 123: `resume` change to Real_RRDBNet weight address that needs to be loaded.
- line 124: `resume_d` change to Discriminator weight address that needs to be loaded.
- line 125: `resume_g` change to Generator weight address that needs to be loaded.

### Result

Source of original paper results: [https://arxiv.org/pdf/2107.10833v2.pdf](https://arxiv.org/pdf/2107.10833v2.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale | Real_RRDBNet (PSNR) | Real_ESRGAN (PSNR) |
|:-------:|:-----:|:-------------------:|:------------------:|
|  Set5   |   4   |    -(**29.28**)     |    -(**26.50**)    |
|  Set14  |   4   |    -(**26.95**)     |    -(**25.23**)    |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png"/></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

_Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan_ <br>

**Abstract** <br>
Though many attempts have been made in blind super-resolution to restore low-resolution images with unknown and complex
degradations, they are still far from addressing general real-world degraded images. In this work, we extend the
powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.
Specifically, a high-order degradation modeling process is introduced to better simulate complex real-world
degradations. We also consider the common ringing and overshoot artifacts in the synthesis process. In addition, we
employ a U-Net discriminator with spectral normalization to increase discriminator capability and stabilize the training
dynamics. Extensive comparisons have shown its superior visual performance than prior works on various real datasets. We
also provide efficient implementations to synthesize training pairs on the fly.
at [this https URL](https://github.com/xinntao/ESRGAN).

[[Paper]](https://arxiv.org/pdf/1609.04802) [[Author implement(PyTorch)]](https://github.com/xinntao/Real-ESRGAN)

```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```
