# -*- coding: utf-8 -*-
#
# @File:   datasets.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:29:53
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-08-22 16:29:28
# @Email:  root@haozhexie.com

import json
import numpy as np
import os
import torch

import utils.io
import utils.transforms

from tqdm import tqdm


def get_dataset(cfg, dataset_name, split):
    if dataset_name == "GOOGLE_EARTH":
        return GoogleEarthDataset(cfg, split)
    elif dataset_name == "GOOGLE_EARTH_BLDG":
        return GoogleEarthBuildingDataset(cfg, split)
    elif dataset_name == "CITY_SAMPLE":
        return CitySampleDataset(cfg, split)
    elif dataset_name == "CITY_SAMPLE_BLDG":
        return CitySampleBuildingDataset(cfg, split)
    elif dataset_name == "CITY_SAMPLE_CAR":
        raise NotImplementedError
    else:
        raise Exception("Unknown dataset: %s" % dataset_name)


def collate_fn(batch):
    data = {}
    for sample in batch:
        for k, v in sample.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        if type(v[0]) == torch.Tensor:
            data[k] = torch.stack(v, 0)
        else:
            data[k] = v

    return data


class CityDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, inst=None):
        super(CityDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.inst = inst
        self.memcached = {}
        self.renderings = []
        self.n_renderings = 0
        self.transforms = None

    @staticmethod
    def get_instance_renderings(renderings, inst_range=None, index_file=None):
        instance_renderings = []
        if os.path.exists(index_file):
            with open(index_file) as fp:
                instance_renderings = json.load(fp)
        else:
            for r in tqdm(renderings, desc="Checking visible instances ..."):
                data = utils.io.IO.get(r["raycasting"])
                ins_map = data["voxel_id"][..., 0, 0] * data["mask"]
                visible_ins = np.unique(
                    ins_map[np.isin(ins_map, [i for i in range(*inst_range)])]
                )
                if len(visible_ins) > 0:
                    instance_renderings.append(r)

            with open(index_file, "w") as fp:
                json.dump(instance_renderings, fp, indent=2)

        return instance_renderings

    def get_n_classes(self, layout=False):
        if self.inst is None:
            return self.cfg.N_CLASSES
        elif self.inst == "BLDG":
            # In layout mode, FACADE and ROOF are considered as the same class
            return self.cfg.N_CLASSES if layout else self.cfg.BLDG.N_CLASSES
        elif self.inst == "CAR":
            raise NotImplementedError
        else:
            raise ValueError("Unknown mode: %s" % self.inst)

    def get_delimeter(self):
        vol_size = self.get_vol_size()
        return [vol_size, vol_size, self.cfg.MAX_HEIGHT]

    def get_vol_size(self):
        cfg = self.cfg if self.inst is None else self.cfg[self.inst]
        return cfg.VOL_SIZE

    def get_center_offset(self):
        return (self.cfg.VOL_SIZE - self.get_vol_size()) / 2

    def __len__(self):
        return (
            self.n_renderings * self.cfg.N_REPEAT
            if self.split == "train"
            else self.n_renderings
        )

    def __getitem__(self, idx):
        rendering = self.renderings[idx % self.n_renderings]
        data = utils.io.IO.get(rendering["raycasting"])

        # print(data.keys())    # dict_keys(['voxel_id', 'depth2', 'raydirs', 'cam_origin', 'img_center', 'mask'])
        data["td_hf"] = self._get_height_field(rendering["td_hf"], self.cfg)
        data["seg_lyt"] = self._get_seg_layout(rendering["seg_lyt"])
        data["footage"] = self._get_footage_img(rendering["footage"])
        if self.inst is not None:
            data["ftp_stats"] = self._get_ftp_stats(rendering["ftp_stats"])

        data = self.transforms(data)
        return data

    def _get_height_field(self, file_path, cfg):
        if file_path in self.memcached:
            return self.memcached[file_path]

        return np.array(utils.io.IO.get(file_path)) / cfg.MAX_HEIGHT

    def _get_seg_layout(self, file_path):
        if file_path in self.memcached:
            return self.memcached[file_path]

        return np.array(utils.io.IO.get(file_path))

    def _get_ftp_stats(self, file_path):
        if file_path in self.memcached:
            return self.memcached[file_path]

        return utils.io.IO.get(file_path)

    def _get_footage_img(self, file_path):
        img = utils.io.IO.get(file_path)
        return (np.array(img) / 255.0 - 0.5) * 2

    def _pin_memory(self, cfg, files):
        for f in tqdm(files, desc="Loading partial files to RAM"):
            for k, v in f.items():
                if k not in cfg.PIN_MEMORY:
                    continue
                elif v in self.memcached:
                    continue
                elif k == "td_hf":
                    self.memcached[v] = self._get_height_field(v, cfg)
                elif k == "seg_lyt":
                    self.memcached[v] = self._get_seg_layout(v)
                elif k == "ftp_stats":
                    self.memcached[v] = self._get_ftp_stats(v)

    def _get_transformations(
        self,
        cfg,
        bev_crop_size,
        img_size,
        img_crop_size,
        rel_ftp_bbox=None,
        instances=None,
        semantic_classes={},
    ):
        # The transformation libraries can be reused in different datasets
        return {
            "Resize": {
                "callback": "Resize",
                "parameters": {
                    "height": img_size[1],
                    "width": img_size[0],
                },
                "objects": ["footage"],
            },
            "BevCrop": {
                "callback": "BevCrop",
                "parameters": {
                    "height": bev_crop_size,
                    "width": bev_crop_size,
                    "rel_ftp_bbox": rel_ftp_bbox,
                },
                # "img_center" is the center of the BEV image (compatible with CityDreamer)
                # Additional data with keys "img_center", "ftp_stats" used in BevCrop.
                "objects": ["td_hf", "seg_lyt"],
            },
            "RandomCrop": {
                "callback": "RandomCrop",
                "parameters": {
                    "height": img_crop_size[1],
                    "width": img_crop_size[0],
                    "n_min_pixels": cfg.N_MIN_PIXELS,
                },
                "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
            },
            "CenterCrop": {
                "callback": "RandomCrop",
                "parameters": {
                    "height": img_crop_size[1],
                    "width": img_crop_size[0],
                    "mode": "center",
                },
                "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
            },
            "InstanceCrop": {
                "callback": "RandomCrop",
                "parameters": {
                    "height": img_crop_size[1],
                    "width": img_crop_size[0],
                    "mode": "instance",
                },
                "objects": ["voxel_id", "depth2", "raydirs", "footage", "mask"],
            },
            "RandomInstances": (
                {
                    "callback": "RandomInstances",
                    "parameters": {
                        "instances": [
                            i
                            for i in range(
                                instances["inst"]["range"][0],
                                instances["inst"]["range"][1],
                            )
                            if instances["inst"]["cond"](i)
                        ],
                        "cont_instances": instances["cnt_inst"],
                    },
                    # "objects": ["voxel_id",  "mask"],
                }
                if instances is not None
                else None
            ),
            "MaskRaydirs": {
                "callback": "MaskRaydirs",
                "parameters": None,
                # "objects": ["voxel_id", "raydirs", "ins"],
            },
            "InstanceToSemantic": {
                "callback": "InstanceToSemantic",
                "parameters": {
                    "semantic_classes": semantic_classes,
                },
                "objects": ["voxel_id", "seg_lyt"],
            },
            "ToOneHot": {
                "callback": "ToOneHot",
                "parameters": {
                    "n_classes": self.get_n_classes(layout=True),
                },
                "objects": ["seg_lyt"],
            },
            "ToTensor": {
                "callback": "ToTensor",
                "parameters": None,
                "objects": [
                    "td_hf",
                    "seg_lyt",
                    "voxel_id",
                    "depth2",
                    "raydirs",
                    "cam_origin",
                    "footage",
                    "mask",
                ],
            },
        }


class GoogleEarthDataset(CityDataset):
    def __init__(self, cfg, split, inst=None):
        dt_cfg = cfg.DATASETS.GOOGLE_EARTH
        super(GoogleEarthDataset, self).__init__(dt_cfg, split, inst)

        self.renderings = self._get_renderings(dt_cfg, split, inst)
        self.n_renderings = len(self.renderings)
        self.semantic_classes = {
            "BLDG_FACADE": {
                "smtc": 0,
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                },
            },
        }
        self.transforms = self._get_data_transform(
            split,
            self._get_transformations(
                dt_cfg,
                bev_crop_size=dt_cfg.VOL_SIZE,
                img_size=dt_cfg.IMAGE_SIZE,
                img_crop_size=(
                    cfg.TRAIN.GANCRAFT.CROP_SIZE
                    if split == "train"
                    else cfg.TEST.GANCRAFT.CROP_SIZE
                ),
                semantic_classes=self.semantic_classes,
            ),
        )

    def _get_renderings(self, cfg, split, inst):
        trajectories = sorted(os.listdir(cfg.FTG_DIR))
        files = [
            {
                "name": "%s/%02d" % (t, i),
                "td_hf": os.path.join(
                    cfg.OSM_DIR, self._get_trajectory_city(t), "hf.png"
                ),
                "seg_lyt": os.path.join(
                    cfg.OSM_DIR, self._get_trajectory_city(t), "seg.png"
                ),
                "footage": os.path.join(
                    cfg.FTG_DIR, t, "footage", "%s_%02d.jpeg" % (t, i)
                ),
                "raycasting": os.path.join(
                    cfg.FTG_DIR, t, "raycasting", "%s_%02d.pkl" % (t, i)
                ),
                "ftp_stats": os.path.join(cfg.FTG_DIR, t, "%s.pkl" % t),
            }
            for t in trajectories
            for i in range(cfg.N_VIEWS)
        ]
        if cfg.PIN_MEMORY:
            self._pin_memory(cfg, files)
        if inst is not None:
            files = CityDataset.get_instance_renderings(
                files, cfg[inst].INS_RANGE, cfg[inst].INDEX_FILE
            )

        return files if split == "train" else files[-32:]

    def _get_trajectory_city(self, trajectory):
        # Trajectory name example: US-SanFrancisco-Chinatown-R624-A354
        return "-".join(trajectory.split("-")[:2])

    def _get_data_transform(self, split, tr):
        return utils.transforms.Compose(
            [
                tr["BevCrop"],
                tr["RandomCrop" if split == "train" else "CenterCrop"],
                tr["InstanceToSemantic"],
                tr["ToOneHot"],
                tr["ToTensor"],
            ]
        )


class GoogleEarthBuildingDataset(GoogleEarthDataset):
    def __init__(self, cfg, split):
        super(GoogleEarthBuildingDataset, self).__init__(cfg, split, inst="BLDG")

        dt_cfg = cfg.DATASETS.GOOGLE_EARTH
        self.semantic_classes = {
            "BLDG_FACADE": {
                "smtc": dt_cfg.CLASSES["BLDG_FACADE"],
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                    "cond": lambda x: x % 2 == 0,
                },
            },
            "BLDG_ROOF": {
                "smtc": dt_cfg.CLASSES["BLDG_ROOF"],
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                    "cond": lambda x: x % 2 == 1,
                },
            },
        }
        self.transforms = self._get_data_transform(
            split,
            self._get_transformations(
                dt_cfg,
                bev_crop_size=dt_cfg.BLDG.VOL_SIZE,
                img_size=dt_cfg.IMAGE_SIZE,
                img_crop_size=(
                    cfg.TRAIN.GANCRAFT.CROP_SIZE
                    if split == "train"
                    else cfg.TEST.GANCRAFT.CROP_SIZE
                ),
                rel_ftp_bbox=True,
                # `instances` is used for the RandomInstances transformation
                instances={
                    "inst": self.semantic_classes["BLDG_FACADE"]["cond"],
                    # NOTE: The ROOF instance is the prev. to the FACADE instance
                    "cnt_inst": [-1],
                },
                semantic_classes=self.semantic_classes,
            ),
        )

    def _get_data_transform(self, _, tr):
        return utils.transforms.Compose(
            [
                tr["RandomInstances"],
                tr["BevCrop"],
                tr["MaskRaydirs"],
                tr["InstanceCrop"],
                tr["InstanceToSemantic"],
                tr["ToOneHot"],
                tr["ToTensor"],
            ]
        )


class CitySampleDataset(CityDataset):
    def __init__(self, cfg, split, inst=None):
        super(CitySampleDataset, self).__init__(cfg.DATASETS.CITY_SAMPLE, split, inst)

        dt_cfg = cfg.DATASETS.CITY_SAMPLE
        self.semantic_classes = {
            "BLDG_FACADE": {
                "smtc": 0,
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                },
            },
            "CAR": {
                "smtc": 0,
                "cond": {
                    "range": (dt_cfg.CAR.INS_RANGE[0], dt_cfg.CAR.INS_RANGE[1]),
                },
            },
        }
        self.renderings = self._get_renderings(dt_cfg, split, inst)
        self.n_renderings = len(self.renderings)
        self.transforms = self._get_data_transform(
            split,
            self._get_transformations(
                dt_cfg,
                bev_crop_size=dt_cfg.VOL_SIZE,
                img_size=dt_cfg.IMAGE_SIZE,
                img_crop_size=(
                    cfg.TRAIN.GANCRAFT.CROP_SIZE
                    if split == "train"
                    else cfg.TEST.GANCRAFT.CROP_SIZE
                ),
                semantic_classes=self.semantic_classes,
            ),
        )

    def _get_renderings(self, cfg, split, inst):
        cities = ["City%02d" % (i + 1) for i in range(cfg.N_CITIES)]
        files = [
            {
                "name": "%s/%s/%04d" % (c, s, i),
                "td_hf": os.path.join(cfg.DIR, c, "Projections", "REST_TD_HF.png"),
                "seg_lyt": os.path.join(cfg.DIR, c, "Projections", "REST_INS_BEV.png"),
                "footage": os.path.join(
                    cfg.DIR,
                    c,
                    "ColorImage",
                    s,
                    "%sSequence.%04d.jpeg" % (c, i),
                ),
                "raycasting": os.path.join(cfg.DIR, c, "Raycasting", "%04d.pkl" % i),
                "ftp_stats": os.path.join(cfg.DIR, c, "Footprints.pkl"),
            }
            for c in cities
            for i in range(cfg.N_VIEWS)
            for s in cfg.CITY_STYLES
        ]
        if cfg.PIN_MEMORY:
            self._pin_memory(cfg, files)
        if inst is not None:
            files = CityDataset.get_instance_renderings(
                files, cfg[inst].INS_RANGE, cfg[inst].INDEX_FILE
            )

        return (
            files
            if split == "train"
            else (
                files
                if split == "train"
                else [f for i, f in enumerate(files) if i % 500 == 0]
            )
        )

    def _get_data_transform(self, split, tr):
        return utils.transforms.Compose(
            [
                tr["BevCrop"],
                tr["Resize"],
                tr["RandomCrop" if split == "train" else "CenterCrop"],
                tr["InstanceToSemantic"],
                tr["ToOneHot"],
                tr["ToTensor"],
            ]
        )


class CitySampleBuildingDataset(CitySampleDataset):
    def __init__(self, cfg, split):
        super(CitySampleBuildingDataset, self).__init__(cfg, split, inst="BLDG")

        dt_cfg = cfg.DATASETS.CITY_SAMPLE
        self.semantic_classes = {
            "BLDG_FACADE": {
                "smtc": dt_cfg.CLASSES["BLDG_FACADE"],
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                    "cond": lambda x: x % 4 == 0,
                },
            },
            "BLDG_ROOF": {
                "smtc": dt_cfg.CLASSES["BLDG_ROOF"],
                "cond": {
                    "range": (dt_cfg.BLDG.INS_RANGE[0], dt_cfg.BLDG.INS_RANGE[1]),
                    "cond": lambda x: x % 4 == 1,
                },
            },
            "CAR": {
                "smtc": 0,
                "cond": {
                    "range": (dt_cfg.CAR.INS_RANGE[0], dt_cfg.CAR.INS_RANGE[1]),
                },
            },
        }
        self.transforms = self._get_data_transform(
            split,
            self._get_transformations(
                dt_cfg,
                bev_crop_size=dt_cfg.BLDG.VOL_SIZE,
                img_size=dt_cfg.IMAGE_SIZE,
                img_crop_size=(
                    cfg.TRAIN.GANCRAFT.CROP_SIZE
                    if split == "train"
                    else cfg.TEST.GANCRAFT.CROP_SIZE
                ),
                rel_ftp_bbox=False,
                # `instances` is used for the RandomInstances transformation
                instances={
                    "inst": self.semantic_classes["BLDG_FACADE"]["cond"],
                    # NOTE: The ROOF instance is the next to the FACADE instance
                    "cnt_inst": [1],
                },
                semantic_classes=self.semantic_classes,
            ),
        )

    def _get_data_transform(self, _, tr):
        return utils.transforms.Compose(
            [
                tr["RandomInstances"],
                tr["Resize"],
                tr["BevCrop"],
                tr["MaskRaydirs"],
                tr["InstanceCrop"],
                tr["InstanceToSemantic"],
                tr["ToOneHot"],
                tr["ToTensor"],
            ]
        )
