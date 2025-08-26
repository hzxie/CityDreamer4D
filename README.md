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


## Changelog 🔥

- [2025/08/27] The source code is released.
- [2025/01/16] The CityTopia dataset is released.
- [2025/01/15] The repo is created.

## Cite this work 📝

```
@article{xie2025citydreamer4d,
  title     = {Compositional Generative Model of Unbounded 4{D} Cities},
  author    = {Xie, Haozhe and 
               Chen, Zhaoxi and 
               Hong, Fangzhou and 
               Liu, Ziwei},
  journal   = {IEEE Transactions on Pattern Analysis and Nachine Intelligence},
  year      = {2025}
}
```

## Datasets

- [OSM](https://gateway.infinitescript.com/s/OSM)
- [GoogleEarth](https://gateway.infinitescript.com/s/GoogleEarth)
- [CityTopia](https://gateway.infinitescript.com/s/CityTopia)

## Pretrained Models

### GoogleEarth

- [Unbounded Layout Generator](https://gateway.infinitescript.com/?f=LayoutGen.pth)
- [Background Stuff Generator](https://gateway.infinitescript.com/?f=CityDreamer-Bgnd.pth)
- [Building Instance Generator](https://gateway.infinitescript.com/?f=CityDreamer-Fgnd.pth)

### CityTopia

- [Background Stuff Generator](https://gateway.infinitescript.com/?f=CityDreamer4D-BG.pth)
- [Building Instance Generator](https://gateway.infinitescript.com/?f=CityDreamer4D-BLDG.pth)

## Installation 📥

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

## License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/hzxie/CityDreamer4D/blob/master/LICENSE). Redistribution and use should follow this license.
