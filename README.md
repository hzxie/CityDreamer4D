<img src="https://www.infinitescript.com/projects/CityDreamer4D/CityDreamer4D-Logo.webp" height="150px" align="right">

# CityDreamer4D: Compositional Generative Model of Unbounded 4D Cities

[Haozhe Xie](https://haozhexie.com), [Zhaoxi Chen](https://frozenburning.github.io/), [Fangzhou Hong](https://hongfz16.github.io/), [Ziwei Liu](https://liuziwei7.github.io/)

S-Lab, Nanyang Technological University

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=hzxie_CityDreamer4D&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=hzxie_CityDreamer4D)
[![codefactor badge](https://www.codefactor.io/repository/github/hzxie/CityDreamer4D/badge)](https://www.codefactor.io/repository/github/hzxie/CityDreamer4D)
![Counter](https://api.infinitescript.com/badgen/count?name=hzxie/CityDreamer4D)
[![arXiv](https://img.shields.io/badge/arXiv-2501.08983-b31b1b.svg)](https://arxiv.org/abs/2501.08983)
[![YouTube](https://img.shields.io/badge/Spotlight%20Video-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/PF6W0Nd27Tk)

![CityDreamer4D Forward Cam - Daytime](https://github.com/user-attachments/assets/14e63958-ab55-409a-87f7-1d359a8f5dea)


## Changelog🔥

- [2025/09/01] Added training and inference instructions.
- [2025/08/27] Released source code.
- [2025/08/24] CityDreamer4D accepted by TPAMI.
- [2025/01/16] Released the CityTopia dataset.
- [2025/01/15] Repository created.

## Cite this work📝

```
@article{xie2025citydreamer4d,
  title     = {Compositional Generative Model of Unbounded 4{D} Cities},
  author    = {Xie, Haozhe and 
               Chen, Zhaoxi and 
               Hong, Fangzhou and 
               Liu, Ziwei},
  journal   = {IEEE Transactions on Pattern Analysis and Nachine Intelligence},
  doi       = {10.1109/TPAMI.2025.3603078},
  year      = {2025}
}
```

## Datasets📚

- [OSM](https://gateway.infinitescript.com/s/OSM)
- [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth)
- [CityTopia](https://gateway.infinitescript.com/s/CityTopia)

## Pretrained Models🧠

### GoogleEarth

- [Background Stuff Generator](https://gateway.infinitescript.com/?f=CityDreamer-Bgnd.pth)
- [Building Instance Generator](https://gateway.infinitescript.com/?f=CityDreamer-Fgnd.pth)

### CityTopia

- [Background Stuff Generator](https://gateway.infinitescript.com/?f=CityDreamer4D-BG.pth)
- [Building Instance Generator](https://gateway.infinitescript.com/?f=CityDreamer4D-BLDG.pth)
- [Vehicle Instance Generator](https://gateway.infinitescript.com/?f=CityDreamer4D-CAR.pth)

## Installation⚙️

Assume that you have installed [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org) in your Python (or Anaconda) environment.  

The CityDreamer source code is tested in PyTorch 2.4.1 with CUDA 11.8 in Python 3.10. You can use the following command to install PyTorch built on CUDA 11.8.

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

After that, the Python dependencies can be installed as following.

```bash
git clone https://github.com/hzxie/CityDreamer4D
cd CityDreamer4D
CITY_DREAMER_HOME=`pwd`
pip install -r requirements.txt
```

The CUDA extensions can be compiled and installed with the following commands.

```bash
cd $CITY_DREAMER_HOME/extensions
for e in `ls -d */`
do
  cd $CITY_DREAMER_HOME/extensions/$e
  pip install .
done
```

## Inference🚀

For the **GoogleEarth** dataset, 24 GB of VRAM is sufficient (tested on an RTX 3090).
For the **CityTopia** dataset, **at least 48 GB of VRAM** is required (tested on an A6000).

**CityTopia-style Generation**

To generate a CityTopia-style city, first download the CityTopia dataset (CityTopia-Annotations-1080p.zip). Then run:

```bash
python3 scripts/dataset_generator.py --data_dir /path/to/citytopia
python3 scripts/traffic_scenario_generator.py --city City01 --steps 120
python3 scripts/inference.py \
  --dataset CITY_SAMPLE \
  --city_sample_dir /path/to/citytopia/City01 \
  --bg_ckpt /path/to/bg-ckpt.pth \
  --bldg_ckpt /path/to/bldg-ckpt.pth \
  --car_ckpt /path/to/car-ckpt.pth
```

**GoogleEarth-style Generation**

The script also supports generating cities in GoogleEarth style. Make sure you have downloaded the OSM dataset before running:

```bash
python3 scripts/inference.py \
  --dataset GOOGLE_EARTH \
  --city_osm_dir /path/to/osm \
  --bg_ckpt /path/to/bg-ckpt.pth \
  --bldg_ckpt /path/to/bldg-ckpt.pth
```

The generated video will be saved at `output/rendering.mp4`.

## Training🏋️

This section provides instructions for training on the **CityTopia** dataset. For training with the **GoogleEarth** dataset, please refer to the [CityDreamer README](https://github.com/hzxie/CityDreamer).

### Dataset Preparation

To generate a CityTopia-style city, first download the CityTopia dataset (CityTopia-Annotations-1080p.zip). Then run:

```bash
python3 scripts/dataset_generator.py --data_dir /path/to/citytopia
```

### Background Stuff Generator Training

#### Update `config.py`

Make sure the config matches the following lines.

```python
cfg.CONST.DATASET                                = "CITY_SAMPLE"
cfg.NETWORK.GANCRAFT.SKY_ENABLED                 = True
```

#### Launch Training 🚀

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py
```

### Building Instance Generator Training

#### Update `config.py`

Make sure the config matches the following lines.

```python
cfg.CONST.DATASET                                = "CITY_SAMPLE"
cfg.NETWORK.GANCRAFT.STYLE_DIM                   = 256
cfg.NETWORK.GANCRAFT.ENCODER                     = "LOCAL"
cfg.NETWORK.GANCRAFT.ENCODER_OUT_DIM             = 64
cfg.NETWORK.GANCRAFT.POS_EMD                     = "SIN_COS"
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_CORDS        = False
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 0
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 0
cfg.TEST.GANCRAFT.CROP_SIZE                      = (360, 180)
```

#### Launch Training 🚀

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py
```

### Vehicle Instance Generator Training

#### Update `config.py`

Make sure the config matches the following lines.

```python
cfg.CONST.DATASET                                = "CITY_SAMPLE"
cfg.NETWORK.GANCRAFT.STYLE_DIM                   = 256
cfg.NETWORK.GANCRAFT.POS_EMD                     = "SIN_COS"
cfg.TEST.GANCRAFT.CROP_SIZE                      = (360, 180)
```

#### Launch Training 🚀

```bash
torchrun --nnodes=1 --nproc_per_node=8 --standalone run.py
```

## License📄

This project is licensed under [NTU S-Lab License 1.0](https://github.com/hzxie/CityDreamer4D/blob/master/LICENSE). Redistribution and use should follow this license.
