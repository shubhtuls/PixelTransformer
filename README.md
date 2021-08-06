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
Please modify the paths in the [config files](conf/).

## Training
See the sample commands in experiments/s2s.py

## Evaluating
See the sample commands in benchmark/

## Preprocessing Data
Most of the image datasets used correspond to standard [torchvision datasets](https://pytorch.org/vision/stable/datasets.html). The cat dataset used is from [Wu. etal's CVPR 2020 work](https://github.com/elliottwu/unsup3d), and can be downloaded using [their provided script](https://github.com/elliottwu/unsup3d/blob/master/data/download_cat.sh).

To extract SDF values for the ShapeNet experiments, we followed the [preprocessing steps from DISN](https://github.com/laughtervv/DISN) although with some modifications to the extraction file. Please use our [modified preprocessing file](external/DISN/preprocessing/create_point_sdf_fullgrid.py) instead for reproducibility.

