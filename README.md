# PixelTransformer
Code release for the ICML 2021 paper "PixelTransformer: Sample Conditioned Signal Generation".

[Project Page](https://shubhtuls.github.io/PixelTransformer/)

## Installation
Please install pytorch and [pytorch3d](https://github.com/facebookresearch/pytorch3d) before the following steps.

```
pip install hydra-core --upgrade
pip install pytorch-lightning
pip install imageio scikit-image

mkdir external; cd external;
git clone git@github.com:kuangliu/pytorch-cifar.git
# if interested in evaluating CIFAR classification accuracy, please train a Resnet-18 model from this repo
```

## Training
Please see the sample commands in experiments/s2s.py

## Evaluating
Please see the sample commands in benchmark/