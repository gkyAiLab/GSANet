# Gamma-enhanced with spatial attention Network for Efficient High Dynamic Range Imaging
By Fangya Li, Ruipeng Gang, Chenghua Li, Jinjing Li, Sai Ma, Chengming Liu and Yizhen Cao

https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Li_Gamma-Enhanced_Spatial_Attention_Network_for_Efficient_High_Dynamic_Range_Imaging_CVPRW_2022_paper.html

## Overview
Overview of the network:
<div align="center">
  <img src='./images/Network.png'>
</div>
​      


## Getting Started

1. [Dataset](#dataset)
2. [Configuration](#configuration)
3. [How to test](#how-to-test)
4. [How to train](#how-to-train)
5. [Visualization](#visualization)

### Dataset
Register a codalab account and log in, then find the download link on this page:
```
https://codalab.lisn.upsaclay.fr/competitions/1514#participate-get-data
```
#### It is strongly recommended to use the data provided by the competition organizer for training and testing, or you need at least a basic understanding of the competition data. Otherwise, you may not get the desired result.

### Configuration
```
pip install -r requirements.txt
```

### How to test

- Modify `dataroot_LQ` and `pretrain_model_G` (you can also use the pretrained model which is provided in the `./pretrained_model`) in `./codes/options/test/test_HDR.yml`, prepare 'results' folder, then run
```
cd codes
python test.py -opt options/test/test_HDR.yml
```
The test results will be saved to `./results/testset_name`.

### How to train

- Prepare the data. Modify `input_folder` and `save_folder` in `./scripts/extract_subimgs_single.py` and prepare 'experiments' folder, then run
```
cd scripts
python extract_subimgs_single.py
```

- Modify `dataroot_LQ` and `dataroot_GT` in `./codes/options/train/train_HDR.yml`, then run
```
cd codes
python train.py -opt options/train/train_HDR.yml
```
The models and training states will be saved to `./experiments/name`.

### Measure operations and runtime

In `./scripts`, several scripts are available. `calculate_ops_example.py` are provided by the competition organizer for measuring operations and runtime.

### Visualization

In `./scripts`, several scripts are available. `data_io.py` and `metrics.py` are provided by the competition organizer for reading/writing data and evaluation. Based on these codes, I provide a script for visualization by using the tone-mapping provided in `metrics.py`. Modify paths of the data in `./scripts/tonemapped_visualization.py` and run
```
cd scripts
python tonemapped_visualization.py
```
to visualize the images.

## Acknowledgment
The code is inspired by [HDRUNet](https://github.com/chxy95/HDRUNet).
