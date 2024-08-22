# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-08-22 16:50:21
# @Email:  root@haozhexie.com

from easydict import EasyDict

# fmt: off
__C                                             = EasyDict()
cfg                                             = __C

#
# Dataset Config
#
cfg.DATASETS                                     = EasyDict()
cfg.DATASETS.GOOGLE_EARTH                        = EasyDict()
cfg.DATASETS.GOOGLE_EARTH.FTG_DIR                = "./data/google-earth"
cfg.DATASETS.GOOGLE_EARTH.OSM_DIR                = "./data/osm"
cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY             = ["td_hf", "seg_lyt", "ftp_stats"]
cfg.DATASETS.GOOGLE_EARTH.IMAGE_SIZE             = (960, 540)
cfg.DATASETS.GOOGLE_EARTH.N_REPEAT               = 1
cfg.DATASETS.GOOGLE_EARTH.MAX_HEIGHT             = 640
cfg.DATASETS.GOOGLE_EARTH.N_VIEWS                = 60
cfg.DATASETS.GOOGLE_EARTH.N_CLASSES              = 7
cfg.DATASETS.GOOGLE_EARTH.CLASSES                = {"ROAD": 1, "BLDG_FACADE": 2, "BLDG_ROOF": 7}
cfg.DATASETS.GOOGLE_EARTH.N_MIN_PIXELS           = 64
cfg.DATASETS.GOOGLE_EARTH.MIN_INSTANCE           = 10
cfg.DATASETS.GOOGLE_EARTH.VOL_SIZE               = 1536
cfg.DATASETS.GOOGLE_EARTH.BLDG                   = EasyDict()
cfg.DATASETS.GOOGLE_EARTH.BLDG.INDEX_FILE        = "./data/google-earth-bldg.json"
cfg.DATASETS.GOOGLE_EARTH.BLDG.N_CLASSES         = 8
cfg.DATASETS.GOOGLE_EARTH.BLDG.VOL_SIZE          = 672
cfg.DATASETS.GOOGLE_EARTH.BLDG.INS_RANGE         = [10, 65536]
cfg.DATASETS.CITY_SAMPLE                         = EasyDict()
cfg.DATASETS.CITY_SAMPLE.DIR                     = "./data/city-sample"
cfg.DATASETS.CITY_SAMPLE.PIN_MEMORY              = ["td_hf", "seg_lyt", "ftp_stats"]
cfg.DATASETS.CITY_SAMPLE.IMAGE_SIZE              = (960, 540)
cfg.DATASETS.CITY_SAMPLE.N_REPEAT                = 1
cfg.DATASETS.CITY_SAMPLE.MAX_HEIGHT              = 2560
cfg.DATASETS.CITY_SAMPLE.N_CITIES                = 5        # 10
cfg.DATASETS.CITY_SAMPLE.N_VIEWS                 = 3000     # 3000
cfg.DATASETS.CITY_SAMPLE.N_CLASSES               = 9
cfg.DATASETS.CITY_SAMPLE.CLASSES                 = {"ROAD": 1, "CAR": 3, "BLDG_FACADE": 7, "BLDG_ROOF": 8}
cfg.DATASETS.CITY_SAMPLE.N_MIN_PIXELS            = 64
cfg.DATASETS.CITY_SAMPLE.MIN_INSTANCE            = 100
cfg.DATASETS.CITY_SAMPLE.CITY_STYLES             = ["Day"]  # ["Day", "Night"]
cfg.DATASETS.CITY_SAMPLE.VOL_SIZE                = 3072
cfg.DATASETS.CITY_SAMPLE.BLDG                    = EasyDict()
cfg.DATASETS.CITY_SAMPLE.BLDG.INDEX_FILE         = "./data/city-sample-bldg.json"
cfg.DATASETS.CITY_SAMPLE.BLDG.N_CLASSES          = 9
cfg.DATASETS.CITY_SAMPLE.BLDG.VOL_SIZE           = 768
cfg.DATASETS.CITY_SAMPLE.BLDG.INS_RANGE          = [100, 5000]
cfg.DATASETS.CITY_SAMPLE.CAR                     = EasyDict()
cfg.DATASETS.CITY_SAMPLE.CAR.INDEX_FILE          = "./data/city-sample-car.json"
cfg.DATASETS.CITY_SAMPLE.CAR.VOL_SIZE            = 0
cfg.DATASETS.CITY_SAMPLE.CAR.INS_RANGE           = [5000, 16384]

#
# Constants
#
cfg.CONST                                        = EasyDict()
cfg.CONST.EXP_NAME                               = ""
cfg.CONST.N_WORKERS                              = 8
cfg.CONST.NETWORK                                = "GANCraft"
cfg.CONST.DATASET                                = "GOOGLE_EARTH"

#
# Directories
#
cfg.DIR                                          = EasyDict()
cfg.DIR.OUTPUT                                   = "./output"

#
# Memcached
#
cfg.MEMCACHED                                    = EasyDict()
cfg.MEMCACHED.ENABLED                            = False
cfg.MEMCACHED.LIBRARY_PATH                       = "/mnt/lustre/share/pymc/py3"
cfg.MEMCACHED.SERVER_CONFIG                      = "/mnt/lustre/share/memcached_client/server_list.conf"
cfg.MEMCACHED.CLIENT_CONFIG                      = "/mnt/lustre/share/memcached_client/client.conf"

#
# WandB
#
cfg.WANDB                                        = EasyDict()
cfg.WANDB.ENABLED                                = False
cfg.WANDB.PROJECT                                = "Moveable-Feast"
cfg.WANDB.ENTITY                                 = "haozhexie"
cfg.WANDB.MODE                                   = "online"
cfg.WANDB.RUN_ID                                 = None
cfg.WANDB.SYNC_TENSORBOARD                       = False

#
# Network
#
cfg.NETWORK                                      = EasyDict()
# GANCraft
cfg.NETWORK.GANCRAFT                             = EasyDict()
cfg.NETWORK.GANCRAFT.STYLE_DIM                   = None         # Options: None, <Any Positive Integers>
cfg.NETWORK.GANCRAFT.N_SAMPLE_POINTS_PER_RAY     = 24
cfg.NETWORK.GANCRAFT.DIST_SCALE                  = 0.25
cfg.NETWORK.GANCRAFT.ENCODER                     = "GLOBAL"     # Options: "GLOBAL", "LOCAL"
cfg.NETWORK.GANCRAFT.ENCODER_OUT_DIM             = 2
cfg.NETWORK.GANCRAFT.GLOBAL_ENCODER_N_BLOCKS     = 6
cfg.NETWORK.GANCRAFT.LOCAL_ENCODER_NORM          = "GROUP_NORM" # Options: "GROUP_NORM", "BATCH_NORM"
cfg.NETWORK.GANCRAFT.POS_EMD                     = "HASH_GRID"  # Options: "HASH_GRID", "SIN_COS"
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_FEATURES     = True
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_CORDS        = True         # Options: True, False
cfg.NETWORK.GANCRAFT.HASH_GRID_N_LEVELS          = 16
cfg.NETWORK.GANCRAFT.HASH_GRID_LEVEL_DIM         = 8
cfg.NETWORK.GANCRAFT.SIN_COS_FREQ_BENDS          = 10
cfg.NETWORK.GANCRAFT.SKY_ENABLED                 = False
cfg.NETWORK.GANCRAFT.SKY_HIDDEN_DIM              = 256
cfg.NETWORK.GANCRAFT.SKY_OUT_DIM_COLOR           = 64
cfg.NETWORK.GANCRAFT.SKY_GLOBAL_AVGPOOL          = False
cfg.NETWORK.GANCRAFT.SKY_POS_EMD_LEVEL_RAYDIR    = 5
cfg.NETWORK.GANCRAFT.SKY_POS_EMD_INCLUDE_RAYDIR  = True
cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM           = 256
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_SIGMA        = 1
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR        = 64
cfg.NETWORK.GANCRAFT.DIS_N_CHANNEL_BASE          = 128

#
# Train
#
cfg.TRAIN                                        = EasyDict()
# GANCraft
cfg.TRAIN.GANCRAFT                               = EasyDict()
cfg.TRAIN.GANCRAFT.N_EPOCHS                      = 500
cfg.TRAIN.GANCRAFT.CKPT_SAVE_FREQ                = 25
cfg.TRAIN.GANCRAFT.BATCH_SIZE                    = 1
cfg.TRAIN.GANCRAFT.EPS                           = 1e-7
cfg.TRAIN.GANCRAFT.WEIGHT_DECAY                  = 0
cfg.TRAIN.GANCRAFT.BETAS                         = (0., 0.999)
cfg.TRAIN.GANCRAFT.CROP_SIZE                     = (192, 192)
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_MODEL         = "vgg19"
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_LAYERS        = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_WEIGHTS       = [0.125, 0.25, 1.0]
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 10
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 10
cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR               = 0.5
cfg.TRAIN.GANCRAFT.EMA_ENABLED                   = False
cfg.TRAIN.GANCRAFT.EMA_RAMPUP                    = 0.05
cfg.TRAIN.GANCRAFT.EMA_N_RAMPUP_ITERS            = 10000
cfg.TRAIN.GANCRAFT.GENERATOR                     = EasyDict()
cfg.TRAIN.GANCRAFT.GENERATOR.LR                  = 1e-4
cfg.TRAIN.GANCRAFT.DISCRIMINATOR                 = EasyDict()
cfg.TRAIN.GANCRAFT.DISCRIMINATOR.ENABLED         = True
cfg.TRAIN.GANCRAFT.DISCRIMINATOR.LR              = 1e-5
cfg.TRAIN.GANCRAFT.DISCRIMINATOR.N_WARMUP_ITERS  = 100000

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.GANCRAFT                                = EasyDict()
cfg.TEST.GANCRAFT.CROP_SIZE                      = (480, 270)
# fmt: on
