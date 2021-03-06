## Learning to Deblur using Light Field Generated and Real Defocused Images

![License CC BY-NC](https://img.shields.io/badge/License-GNU_AGPv3-yellowgreen.svg?style=flat)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Jvfbn8HnWAmgTKFpU8fW56wXSbe1S2QI?usp=sharing)

<img src="./assets/teaser.png" width="50%" alt="teaser figure">

This repository contains the official PyTorch implementation of the following paper:

> **[Learning to Deblur using Light Field Generated and Real Defocused Images](https://arxiv.org/pdf/2204.00367.pdf)**<br>
> Lingyan Ruan<sup>\*</sup>, Bin Chen<sup>\*</sup>, Jizhou Li, Miuling Lam （\* equal contribution）<br>
> IEEE Computer Vision and Pattern Recognition (**CVPR Oral**) 2022

**[PROJECT PAGE](http://lyruan.com/Projects/DRBNet/index.html)** | **[INTERACTIVE WEB APP](https://xi5tau4hrb3hsakw.anvil.app/FJJ5EACSBF63RE7RQL2K6ZDZ)**

If you find our code useful, please consider citing our paper:

```
{
    Come Soon!
}
```

## Code [Come Soon!!]

### Prerequisites

![Ubuntu](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.8-yellowgreen.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20-yellowgreen.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-yellowgreen.svg?style=plastic)

Notes: the code may also work with other library versions that didn't specify here.

#### 1. Installation

Clone this project to your local machine

```bash
$ git clone https://github.com/lingyanruan/DRBNet.git
$ cd DRBNet
```

#### 2. Pre-trained models

Download and unzip [pretrained weights](placeholder) under `./weight/`:

#### 3. Datasets

Download and unzip test sets ([LFDOF](https://sweb.cityu.edu.hk/miullam/AIFNET/), [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel), [CUHK](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html) and [RealDOF](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1)):

#### 4. Command Line

```shell
python Eval.py
```

## Performance improved on existing works - [DPDNet & KPAC]

You may go for [DPDNet](https://github.com/lingyanruan/DPDNet) and [KPAC-Net](https://github.com/lingyanruan/KPAC-Net) for their improved version. Details could be found in [Why LFDOF?] section (Table 4 & Figure 8) in the main paper. Their original version could be found [Here: DPDNet-scr](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) and [Here: KPAC-Net-scr](https://github.com/lingyanruan/KPAC-Net)

## Relevant Resources

- TCI'20 paper: AIFNet: All-in-focus Image Restoration Network using a Light Field-based Dataset &nbsp; [[Paper](https://ieeexplore.ieee.org/document/9466450)] [[Project page](https://sweb.cityu.edu.hk/miullam/AIFNET/)] [[LFDOF Dataset](https://sweb.cityu.edu.hk/miullam/AIFNET/)] [[Code](https://github.com/HyeongseokSon1/KPAC)]

## Contact

Should you have any questions, please open an issue or contact me [lyruanruan@gmail.com](mailto:lyruanruan@gmail.com)

## License

This software is being made available under the terms in the [LICENSE](LICENSE) file.
