# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2023-05-31 15:01:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-08-27 04:51:39
# @Email:  root@haozhexie.com

import argparse
import copy
import cv2
import logging
import math
import numpy as np
import os
import pickle
import torch
import torchvision.transforms
import sys

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.footprint_extruder
import extensions.voxlib
import models.gancraft
import scripts.traffic_scenario_generator
import utils.datasets
import utils.helpers


def get_cfg_value(key, dataset=None):
    assert dataset in ["GOOGLE_EARTH", "CITY_SAMPLE", None]
    CONSTANTS = {
        "IMAGE_HEIGHT": 540,
        "IMAGE_WIDTH": 960,
        "IMAGE_PADDING": 8,
        "BLDG_ROOF_HEIGHT": 1,
        "N_VOXEL_SAMPLES": 6,
        "N_VIEWPOINTS": 24,
        "RAYCAST_CACHE_DIR": "/tmp/raycast",
        "BLDG_INST_RANGE": [10, 30000],
        "CAR_INST_RANGE": [30000, 32767],
        "N_CAR_CLASSES": 7,
    }

    if key in CONSTANTS:
        return CONSTANTS[key]
    else:
        return _get_dataset_cfg_value(key, dataset)


def _get_dataset_cfg_value(key, dataset):
    from config import cfg

    CFG_KEYS = {
        "N_LAYOUT_CLASSES": "N_CLASSES",
        "CLASSES": "CLASSES",
        "BLDG_MAX_HEIGHT": "MAX_HEIGHT",
        "CAR_MAX_HEIGHT": "MAX_HEIGHT",
        "VOL_SIZE/LAYOUT": "VOL_SIZE",
        "VOL_SIZE/BLDG": "BLDG.VOL_SIZE",
        "VOL_SIZE/CAR": "CAR.VOL_SIZE",
    }
    # The constants are not defined in config.py but aligned with dataset_generator.py
    CFG_VALUES = {
        "IMAGE_VFOV": {
            "GOOGLE_EARTH": 10.019869883021967,
            "CITY_SAMPLE": 10.809306586025498,
        },
        "BLDG_ROOF_OFFSET": {"GOOGLE_EARTH": -1, "CITY_SAMPLE": 1},
        "BLDG_INST_MULTIPLIER": {"GOOGLE_EARTH": 2, "CITY_SAMPLE": 4},
    }
    if key in CFG_KEYS:
        # Read the value from the dataset config recursively
        path = CFG_KEYS[key].split(".")
        _cfg = cfg.DATASETS[dataset]
        for p in path:
            if p not in _cfg:
                return None
            _cfg = getattr(_cfg, p)

        return _cfg
    elif key in CFG_VALUES:
        return CFG_VALUES[key][dataset]
    elif key == "VOL_SIZE/EXTEND":
        return _get_dataset_cfg_value(
            "VOL_SIZE/LAYOUT", dataset
        ) + 2 * _get_dataset_cfg_value("VOL_SIZE/BLDG", dataset)
    return None


def _get_compatible_cfg(cfg, dataset):
    inst = dataset.split("_")[-1]
    inst = inst if inst in ["BLDG", "CAR"] else None
    dataset = "_".join(dataset.split("_")[:-1]) if inst is not None else dataset
    assert dataset in ["GOOGLE_EARTH", "CITY_SAMPLE"], "Unknown dataset: %s" % dataset
    dt_cfg = cfg.DATASETS[dataset]
    model_cfg = cfg.NETWORK.GANCRAFT

    # To make it compatible with the released CityDreamer pretrained models
    # The "GOOGLE_EARTH_BUILDING" only appears in the legacy config file
    if dataset == "GOOGLE_EARTH" and "GOOGLE_EARTH_BUILDING" in cfg.DATASETS:
        DEPRECATED_MODLE_CFG_KEYS = [
            "BUILDING_MODE",
            "HASH_GRID_RESOLUTION",
            "CENTER_OFFSET",
            "NORMALIZE_DELIMETER",
            "N_CLASSES",
        ]
        from config import cfg as _cfg

        default_cfg = copy.deepcopy(_cfg)
        # Overwrite the legacy config from the latest config file
        dt_cfg = default_cfg.DATASETS.GOOGLE_EARTH
        legacy_model_cfg, model_cfg = model_cfg, default_cfg.NETWORK.GANCRAFT
        # Disable sky network for the CityDreamer pretrained models
        model_cfg.SKY_ENABLED = False
        # Overwrite the default config with the legacy config
        for k, v in legacy_model_cfg.items():
            if k in DEPRECATED_MODLE_CFG_KEYS:
                continue
            elif v != model_cfg[k]:
                model_cfg[k] = v

    return dt_cfg, model_cfg, inst


def _get_model(dataset, ckpt_file_path):
    if not os.path.exists(ckpt_file_path):
        return None

    ckpt = torch.load(ckpt_file_path, weights_only=False)
    dt_cfg, model_cfg, inst = _get_compatible_cfg(ckpt["cfg"], dataset)
    city_dataset = utils.datasets.CityDataset(dt_cfg, None, inst)
    model = models.gancraft.GanCraftGenerator(
        cfg=model_cfg,
        n_classes={
            "SMT": city_dataset.get_n_classes(),
            "LYT": city_dataset.get_n_classes(layout=True),
        },
        delimeter=city_dataset.get_delimeter(),
        vol_size=city_dataset.get_vol_size(),
        center_offset=city_dataset.get_center_offset(),
    )
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.output_device = torch.device("cpu")

    model.load_state_dict(ckpt["gancraft_g"], strict=False)
    return model


def get_models(dataset, bg_ckpt, bldg_ckpt, car_ckpt):
    bg_model = _get_model(dataset, bg_ckpt)

    bldg_model = None
    if bldg_ckpt is not None:
        bldg_model = _get_model("%s_BLDG" % dataset, bldg_ckpt)

    car_model = None
    if car_ckpt is not None:
        car_model = _get_model("%s_CAR" % dataset, car_ckpt)
        if car_model is not None:
            car_model.module.center_offset = 0

    return bg_model, bldg_model, car_model


def get_city_layout(
    dataset,
    dataset_dirs,
    bldg_facade_cid,
    bldg_inst_mult,
    bldg_inst_range,
    bldg_max_height,
):
    if dataset == "CITY_SAMPLE":
        return get_city_sample_layout(dataset_dirs["city_sample"], bldg_inst_range)
    elif dataset == "GOOGLE_EARTH":
        return get_osm_city_layout(
            dataset_dirs["osm"],
            bldg_facade_cid,
            bldg_inst_mult,
            bldg_inst_range,
            bldg_max_height,
        )
    else:
        raise ValueError("Unknown dataset: %s" % dataset)


def _get_projections(projection_dir, layers, maps):
    projections = {}
    for k in layers:
        projections[k] = {
            mk: np.array(
                Image.open(os.path.join(projection_dir, "%s_%s.png" % (k, mk)))
            ).astype(np.int16)
            for mk in maps
        }

    return projections


def get_city_sample_layout(city_sample_dir, bldg_inst_range):
    # NOTE: Static CARS are omitted in the CITY_SAMPLE dataset
    projections = _get_projections(
        os.path.join(city_sample_dir, "Projections"),
        ["FREEWAY", "REST"],
        ["INS_BEV", "TD_HF", "BU_HF"],
    )

    with open(os.path.join(city_sample_dir, "Footprints.pkl"), "rb") as fp:
        ftp_stats = pickle.load(fp)
        ftp_stats = {
            k: v
            for k, v in ftp_stats.items()
            if k >= bldg_inst_range[0] and k < bldg_inst_range[1]
        }

    return projections, ftp_stats


def get_osm_city_layout(
    city_osm_dir, bldg_facade_cid, bldg_inst_mult, bldg_inst_range, bldg_max_height
):
    hf = np.array(Image.open(os.path.join(city_osm_dir, "hf.png")))
    seg = np.array(Image.open(os.path.join(city_osm_dir, "seg.png")).convert("P"))
    ins_seg, bldg_stats = _get_instance_seg_layout(
        seg, bldg_facade_cid, bldg_inst_mult, bldg_inst_range
    )
    hf = _clip_height_field(hf, bldg_max_height).astype(np.int16)

    projections = {
        "REST": {
            "TD_HF": hf,
            "BU_HF": np.zeros_like(hf),
            "INS_BEV": ins_seg.astype(np.int32),
        }
    }
    return projections, bldg_stats


def _get_instance_seg_layout(
    seg_layout, bldg_facade_cid, bldg_inst_mult, bldg_inst_range
):
    OSM_CLASSES = {"CONSTRUCTION": 4}
    # Mapping constructions to buildings
    seg_layout[seg_layout == OSM_CLASSES["CONSTRUCTION"]] = bldg_facade_cid
    # Generate building instance seg maps
    # https://github.com/hzxie/CityDreamer/blob/master/scripts/dataset_generator.py#L393
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (seg_layout == bldg_facade_cid).astype(np.uint8), connectivity=4
    )
    # Remove non-building instance masks
    labels[seg_layout != bldg_facade_cid] = 0
    # Building instance mask
    building_mask = labels != 0
    # Make building instance IDs are even numbers and start from min_bldg_inst
    # Assume the ID of a facade instance is 2k (4k), the corresponding roof instance is 2k-1 (4k+1).
    labels = (labels + bldg_inst_range[0]) * bldg_inst_mult
    # Ignore the instances that are out of the range
    labels[labels >= bldg_inst_range[1]] = 0

    seg_layout[seg_layout == bldg_facade_cid] = 0
    seg_layout = seg_layout * (1 - building_mask) + labels * building_mask
    assert np.max(labels) < bldg_inst_range[1]

    bldg_stats = {
        (i + bldg_inst_range[0]) * bldg_inst_mult: s[:4] for i, s in enumerate(stats)
    }
    # Ignore the instances that are out of the range
    bldg_stats = {k: v for k, v in bldg_stats.items() if k < bldg_inst_range[1]}

    return seg_layout.astype(np.int32), bldg_stats


def _clip_height_field(hf, layout_max_height):
    hf[hf >= layout_max_height] = layout_max_height - 1
    return hf


def get_traffic_scenarios(dataset, dataset_dirs):
    if dataset == "CITY_SAMPLE":
        scenario_dir = os.path.join(dataset_dirs["city_sample"], "Scenarios")
    elif dataset == "GOOGLE_EARTH":
        raise NotImplementedError
    else:
        raise ValueError("Unknown dataset: %s" % dataset)

    scenarios = []
    for f in tqdm(sorted([f for f in os.listdir(scenario_dir) if f.endswith(".pkl")])):
        with open(os.path.join(scenario_dir, f), "rb") as fp:
            scenario = pickle.load(fp)

        scenarios.append(scenario)
    return scenarios


def get_latent_codes(bldg_stats, scenarios, bg_style_dim, output_device):
    bg_z = _get_z(output_device, bg_style_dim)
    building_zs = {k: _get_z(output_device) for k in bldg_stats.keys()}
    car_zs = None
    if scenarios:
        car_zs = {
            k: _get_z(output_device)
            for k in set(
                [
                    t["id"]
                    for scene in scenarios
                    for metadata in scene["METADATA"].values()
                    for t in metadata["TRACK"]
                ]
            )
        }
    return bg_z, building_zs, car_zs


def _get_z(device, z_dim=256):
    if z_dim is None:
        return None

    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


def _get_image_patch(image, tl, br):
    return image[tl[1] : br[1], tl[0] : br[0]]


def _get_bev_map_bbox(cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return {"TL": np.array([sx, sy]), "BR": np.array([ex, ey])}


def _bldg_class_callback(seg, cfg, _):
    seg[(seg >= cfg["INST_RANGE"][0]) & (seg < cfg["INST_RANGE"][1])] = cfg[
        "FACADE_CID"
    ]
    return seg


def get_hf_seg_tensor(projections, cx, cy, n_classes, cfg, output_device):
    bev_map_bbox = None
    if cx is not None and cy is not None:
        bev_map_bbox = _get_bev_map_bbox(cx, cy, cfg["VOL_SIZE"])

    part_hf = projections["TD_HF"]
    if bev_map_bbox is not None:
        part_hf = _get_image_patch(
            projections["TD_HF"], bev_map_bbox["TL"], bev_map_bbox["BR"]
        )
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(output_device)
    part_hf = part_hf / cfg["MAX_HEIGHT"]

    part_seg = projections["INS_BEV"]
    if bev_map_bbox is not None:
        part_seg = _get_image_patch(
            projections["INS_BEV"], bev_map_bbox["TL"], bev_map_bbox["BR"]
        )
    instances = np.unique(part_seg)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(output_device)
    if "CLASS_MAPPER" in cfg:
        part_seg = cfg["CLASS_MAPPER"](part_seg, cfg, n_classes)

    part_seg = utils.helpers.masks_to_onehots(part_seg[:, 0, :, :], n_classes)
    return torch.cat([part_hf, part_seg], dim=1), instances


def get_part_bldg_stats(instances, bldg_stats, cx, cy, bldg_inst_range):
    _buildings = instances[
        (instances > bldg_inst_range[0]) & (instances < bldg_inst_range[1])
    ]
    _bldg_stats = {}
    for b in _buildings:
        # Skip the duplicated building instances
        if b in _bldg_stats:
            continue

        _bldg_stats[b] = [
            bldg_stats[b][1] - cy + bldg_stats[b][3] / 2,
            bldg_stats[b][0] - cx + bldg_stats[b][2] / 2,
        ]

    return _bldg_stats


def get_part_car_stats(instances, scenario, cx, cy, car_inst_range):
    cars = instances[(instances >= car_inst_range[0]) & (instances < car_inst_range[1])]
    car_stats = {}
    for values in scenario.values():
        for s in values["TRACK"]:
            if s["id"] in cars:
                car_stats[s["id"]] = s
                s["cy"] -= cy
                s["cx"] -= cx
                s["cz"] = s["cz"] if "cz" in s else 0
                # In traffic scenario, the 0 degree is the east direction,
                # 90 degree is the south direction. In NeRF, the 0 degree
                # is the north direction.
                s["heading"] = (s["heading"] + 90) % 360

    return car_stats


def get_seg_volume(projections, traffic_scenario, cx, cy, vol_sizes, bldg_cfg):
    bev_map_bbox = _get_bev_map_bbox(cx, cy, vol_sizes["EXT"])
    seg_volume = torch.zeros(
        (vol_sizes["LAYOUT"], vol_sizes["LAYOUT"], bldg_cfg["MAX_HEIGHT"]),
        dtype=(
            torch.int16
            if projections["REST"]["INS_BEV"].dtype == np.int16
            else torch.int32
        ),
        device=torch.device("cuda:0"),
    )
    for proj in [projections, traffic_scenario]:
        if proj is None:
            continue

        _projections = {}
        for layer in proj.values():
            for mk in ["TD_HF", "BU_HF", "INS_BEV"]:
                _projections[mk] = _get_image_patch(
                    layer[mk],
                    bev_map_bbox["TL"] + vol_sizes["BLDG"],
                    bev_map_bbox["BR"] - vol_sizes["BLDG"],
                )
                assert _projections[mk].shape == (
                    vol_sizes["LAYOUT"],
                    vol_sizes["LAYOUT"],
                )

            assert np.min(_projections["TD_HF"]) >= 0
            assert np.max(_projections["TD_HF"]) < bldg_cfg["MAX_HEIGHT"]
            seg_volume = extensions.footprint_extruder.extrude_footprint(
                seg_volume,
                torch.from_numpy(_projections["INS_BEV"]).to(seg_volume.device),
                torch.from_numpy(_projections["TD_HF"]).to(seg_volume.device),
                torch.from_numpy(_projections["BU_HF"]).to(seg_volume.device),
                0,
                bldg_cfg["ROOF_HEIGHT"],
                0,
                bldg_cfg["ROOF_OFFSET"],
                bldg_cfg["INST_RANGE"][0],
                bldg_cfg["INST_RANGE"][1],
            )

    logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    return seg_volume


def get_orbit_camera_positions(radius, altitude, vol_size_layout, n_viewpoints):
    # TODO: More flexible for CITY_SAMPLE
    cam_look_at = {"x": vol_size_layout // 2 - 1, "y": vol_size_layout // 2 - 1, "z": 0}
    camera_positions = []
    cx = vol_size_layout // 2
    cy = cx
    for i in range(n_viewpoints):
        theta = 2 * math.pi / n_viewpoints * i
        cam_x = cx + radius * math.cos(theta)
        cam_y = cy + radius * math.sin(theta)
        camera_positions.append(
            {
                "cam_position": {"x": cam_x, "y": cam_y, "z": altitude},
                "cam_look_at": cam_look_at,
            }
        )
    return camera_positions


def get_voxel_intersection_perspective(
    seg_volume, cam_pose, img_sizes, img_vfov, n_voxel_samples
):
    CAMERA_FOCAL = img_sizes["HEIGHT"] / 2 / np.tan(np.deg2rad(img_vfov))
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    cam_origin = torch.tensor(
        [
            cam_pose["cam_position"]["y"],
            cam_pose["cam_position"]["x"],
            cam_pose["cam_position"]["z"],
        ],
        dtype=torch.float32,
        device=seg_volume.device,
    )
    viewdir = torch.tensor(
        [
            cam_pose["cam_look_at"]["y"] - cam_pose["cam_position"]["y"],
            cam_pose["cam_look_at"]["x"] - cam_pose["cam_position"]["x"],
            cam_pose["cam_look_at"]["z"] - cam_pose["cam_position"]["z"],
        ],
        dtype=torch.float32,
        device=seg_volume.device,
    )
    voxel_id, depth2, raydirs = extensions.voxlib.ray_voxel_intersection_perspective(
        seg_volume,
        cam_origin,
        viewdir,
        torch.tensor([0, 0, 1], dtype=torch.float32),
        CAMERA_FOCAL,
        [
            img_sizes["HEIGHT"] / 2,
            img_sizes["WIDTH"] / 2,
        ],
        [img_sizes["HEIGHT"], img_sizes["WIDTH"]],
        n_voxel_samples,
    )
    return (
        voxel_id.unsqueeze(dim=0),
        depth2.permute(1, 2, 0, 3, 4).unsqueeze(dim=0),
        raydirs.unsqueeze(dim=0),
        cam_origin.unsqueeze(dim=0),
    )


def get_pad_img_bbox(sx, ex, sy, ey, img_height, img_width, img_padding):
    psx = sx - img_padding if sx != 0 else 0
    psy = sy - img_padding if sy != 0 else 0
    pex = ex + img_padding if ex != img_width else img_width
    pey = ey + img_padding if ey != img_height else img_height
    return psx, pex, psy, pey


def get_img_without_pad(img, sx, ex, sy, ey, psx, pex, psy, pey, img_padding):
    if img_padding == 0:
        return img

    return img[
        :,
        :,
        sy - psy : ey - pey if ey != pey else ey,
        sx - psx : ex - pex if ex != pex else ex,
    ]


def render(
    patch_size, hf_segs, raycast, models, stats, zs, vol_sizes, img_cfg, other_cfg
):
    bldg_seg = raycast["VOXEL_ID"][0, :, :, 0, 0]
    buildings = torch.unique(
        bldg_seg[
            (bldg_seg >= other_cfg["BLDG_INST_RANGE"][0])
            & (bldg_seg < other_cfg["BLDG_INST_RANGE"][1])
        ]
    )
    buildings = buildings[buildings % other_cfg["BLDG_INST_MULTIPLIER"] == 0]
    # Fix: Roof is visible but the facade is not visible (Hard-coded for CITY_SAMPLE)
    # bldg_roofs = buildings[buildings % other_cfg["BLDG_INST_MULTIPLIER"] == 1]
    # bldg_facades0 = bldg_roofs - 1
    # bldg_facades1 = buildings[buildings % other_cfg["BLDG_INST_MULTIPLIER"] == 0]
    # buildings = torch.unique(torch.cat([bldg_facades0, bldg_facades1]))

    cars = []
    if stats["CAR"] is not None:
        car_seg = raycast["VOXEL_ID"][0, :, :, 0, 0]
        cars = torch.unique(
            car_seg[
                (car_seg >= other_cfg["CAR_INST_RANGE"][0])
                & (car_seg < other_cfg["CAR_INST_RANGE"][1])
            ]
        )

    with torch.no_grad():
        bg_img = render_bg(
            patch_size,
            models["BG"],
            hf_segs["STATIC"],
            raycast["VOXEL_ID"],
            raycast["DEPTH2"],
            raycast["RAYDIRS"],
            raycast["CAM_ORIGIN"],
            zs["BG"],
            vol_sizes,
            img_cfg,
            other_cfg,
        )
        for b in buildings:
            bldg_img, bldg_mask = render_bldg(
                patch_size,
                models["BLDG"],
                b.item(),
                hf_segs["STATIC"],
                raycast["VOXEL_ID"],
                raycast["DEPTH2"],
                raycast["RAYDIRS"],
                raycast["CAM_ORIGIN"],
                stats["BLDG"][b.item()],
                zs["BLDG"][b.item()],
                vol_sizes,
                img_cfg,
                other_cfg,
            )
            bg_img = bg_img * (1 - bldg_mask) + bldg_img * bldg_mask

        for c in cars:
            car_img, car_mask = render_car(
                patch_size,
                models["CAR"],
                c.item(),
                raycast["VOXEL_ID"],
                raycast["CAM_ORIGIN"],
                stats["CAR"][c.item()],
                zs["CAR"][c.item()],
                vol_sizes,
                img_cfg,
                other_cfg,
            )
            bg_img = bg_img * (1 - car_mask) + car_img * car_mask

    return bg_img


def render_bg(
    patch_size,
    bg_model,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_origin,
    z,
    vol_sizes,
    img_cfg,
    inst_cfg,
):
    assert hf_seg.size(2) == vol_sizes["EXT"]
    assert hf_seg.size(3) == vol_sizes["EXT"]
    hf_seg = hf_seg[
        :,
        :,
        vol_sizes["BLDG"] : -vol_sizes["BLDG"],
        vol_sizes["BLDG"] : -vol_sizes["BLDG"],
    ]
    assert hf_seg.size(2) == vol_sizes["LAYOUT"]
    assert hf_seg.size(3) == vol_sizes["LAYOUT"]

    blurrer = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2, 2))
    _voxel_id = copy.deepcopy(voxel_id)
    # Maps all building instances to the same facade ID
    _voxel_id[voxel_id >= inst_cfg["CAR_INST_RANGE"][0]] = inst_cfg["ROAD_CID"]
    _voxel_id[voxel_id >= inst_cfg["BLDG_INST_RANGE"][0]] = inst_cfg["FACADE_CID"]
    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    bg_img = torch.zeros(
        1,
        3,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=bg_model.output_device,
    )
    # bg_img[bg_img == 0] = -1
    # Render patch by patch to avoid OOM
    for i in range(img_cfg["HEIGHT"] // patch_size[0]):
        for j in range(img_cfg["WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(
                sx, ex, sy, ey, img_cfg["HEIGHT"], img_cfg["WIDTH"], img_cfg["PADDING"]
            )
            output, _ = bg_model(
                hf_seg=hf_seg,
                voxel_id=_voxel_id[:, psy:pey, psx:pex],
                depth2=depth2[:, psy:pey, psx:pex],
                raydirs=raydirs[:, psy:pey, psx:pex],
                cam_origin=cam_origin,
                ftp_stats=None,
                z=z,
                deterministic=True,
            )
            # Make road blurry
            road_mask = (
                (_voxel_id[:, None, psy:pey, psx:pex, 0, 0] == inst_cfg["ROAD_CID"])
                .repeat(1, 3, 1, 1)
                .float()
            )
            output = blurrer(output) * road_mask + output * (1 - road_mask)
            bg_img[:, :, sy:ey, sx:ex] = get_img_without_pad(
                output, sx, ex, sy, ey, psx, pex, psy, pey, img_cfg["PADDING"]
            )

    return bg_img


def render_bldg(
    patch_size,
    bldg_model,
    curr_bldg_inst,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_origin,
    bldg_stats,
    building_z,
    vol_sizes,
    img_cfg,
    inst_cfg,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _curr_bldg = torch.tensor(
        [curr_bldg_inst, curr_bldg_inst + inst_cfg["ROOF_OFFSET"]],
        device=voxel_id.device,
    )
    _voxel_id[~torch.isin(_voxel_id, _curr_bldg)] = 0
    _voxel_id[voxel_id == curr_bldg_inst] = inst_cfg["FACADE_CID"]
    _voxel_id[voxel_id == curr_bldg_inst + inst_cfg["ROOF_OFFSET"]] = inst_cfg[
        "ROOF_CID"
    ]
    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[_voxel_id[..., 0, 0] == 0] = 0

    # Crop the "hf_seg" image using the center of the target building as the reference
    cx = vol_sizes["EXT"] // 2 + int(bldg_stats[1])
    cy = vol_sizes["EXT"] // 2 + int(bldg_stats[0])
    sx = cx - vol_sizes["BLDG"] // 2
    ex = cx + vol_sizes["BLDG"] // 2
    sy = cy - vol_sizes["BLDG"] // 2
    ey = cy + vol_sizes["BLDG"] // 2
    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg = hf_seg[:, :, sy:ey, sx:ex]

    bldg_img = torch.zeros(
        1,
        3,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=bldg_model.output_device,
    )
    bldg_mask = torch.zeros(
        1,
        1,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=bldg_model.output_device,
    )
    # Prevent some buildings are out of bound.
    # THIS SHOULD NEVER HAPPEN AGAIN.
    # if (
    #     _hf_seg.size(2) != vol_sizes["BLDG"]
    #     or _hf_seg.size(3) != vol_sizes["BLDG"]
    # ):
    #     return fg_img, fg_mask

    # Render patch by patch to avoid OOM
    for i in range(img_cfg["HEIGHT"] // patch_size[0]):
        for j in range(img_cfg["WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(
                sx, ex, sy, ey, img_cfg["HEIGHT"], img_cfg["WIDTH"], img_cfg["PADDING"]
            )
            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex]) > 0:
                output, _ = bldg_model(
                    _hf_seg,
                    _voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    _raydirs[:, psy:pey, psx:pex],
                    cam_origin,
                    ftp_stats=torch.from_numpy(np.array(bldg_stats)).unsqueeze(dim=0),
                    z=building_z,
                    deterministic=True,
                )
                facade_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == curr_bldg_inst
                ).unsqueeze(dim=1)
                roof_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0]
                    == curr_bldg_inst + inst_cfg["ROOF_OFFSET"]
                ).unsqueeze(dim=1)
                facade_img = facade_mask * get_img_without_pad(
                    output, sx, ex, sy, ey, psx, pex, psy, pey, img_cfg["PADDING"]
                )
                # Make roof blurry
                # output_fg = F.interpolate(
                #     F.interpolate(output_fg * 0.8, scale_factor=0.75),
                #     scale_factor=4 / 3,
                # ),
                roof_img = roof_mask * get_img_without_pad(
                    output,
                    sx,
                    ex,
                    sy,
                    ey,
                    psx,
                    pex,
                    psy,
                    pey,
                    img_cfg["PADDING"],
                )
                bldg_mask[:, :, sy:ey, sx:ex] = torch.logical_or(facade_mask, roof_mask)
                bldg_img[:, :, sy:ey, sx:ex] = (
                    facade_img * facade_mask + roof_img * roof_mask
                )

    return bldg_img, bldg_mask


def render_car(
    patch_size,
    car_model,
    curr_car_inst,
    g_voxel_id,
    g_cam_origin,
    car_stats,
    car_z,
    vol_sizes,
    img_cfg,
    other_cfg,
):
    g_mask = g_voxel_id[..., 0, 0] == curr_car_inst
    # Convert to a local coordinate system of size 32^3
    offsets = {
        "x": -vol_sizes["LAYOUT"] // 2 - car_stats["cx"] + vol_sizes["CAR"] // 2,
        "y": -vol_sizes["LAYOUT"] // 2 - car_stats["cy"] + vol_sizes["CAR"] // 2,
        "z": -car_stats["cz"],
    }
    g_cam_look_at = {
        "x": vol_sizes["LAYOUT"] // 2 - 1,
        "y": vol_sizes["LAYOUT"] // 2 - 1,
        "z": 0,
    }
    cam_pose = {
        "cam_position": {
            "x": g_cam_origin[0][1] + offsets["x"],
            "y": g_cam_origin[0][0] + offsets["y"],
            "z": g_cam_origin[0][2] + offsets["z"],
        },
        "cam_look_at": {
            "x": g_cam_look_at["x"] + offsets["x"],
            "y": g_cam_look_at["y"] + offsets["y"],
            "z": g_cam_look_at["z"] + offsets["z"],
        },
    }
    pivot = {"x": vol_sizes["CAR"] // 2, "y": vol_sizes["CAR"] // 2}
    cam_pose["cam_position"] = _get_rotated_coordinates(
        cam_pose["cam_position"], pivot, car_stats["heading"]
    )
    cam_pose["cam_look_at"] = _get_rotated_coordinates(
        cam_pose["cam_look_at"], pivot, car_stats["heading"]
    )

    hf_seg, volume = _get_car_volume(
        curr_car_inst,
        other_cfg["N_CAR_CLASSES"],
        other_cfg["CAR_BEV_FILE"],
        vol_sizes["CAR"],
        car_model.output_device,
    )
    voxel_id, depth2, raydirs, cam_origin = get_voxel_intersection_perspective(
        volume,
        cam_pose,
        img_cfg,
        img_cfg["VFOV"],
        other_cfg["N_VOXEL_SAMPLES"],
    )

    car_img = torch.zeros(
        1,
        3,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=car_model.output_device,
    )
    car_mask = torch.zeros(
        1,
        1,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=car_model.output_device,
    )
    # Render patch by patch to avoid OOM
    for i in range(img_cfg["HEIGHT"] // patch_size[0]):
        for j in range(img_cfg["WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(
                sx, ex, sy, ey, img_cfg["HEIGHT"], img_cfg["WIDTH"], img_cfg["PADDING"]
            )
            if torch.count_nonzero(g_mask[:, sy:ey, sx:ex]) > 0:
                color, sigma = car_model(
                    hf_seg,
                    voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    raydirs[:, psy:pey, psx:pex],
                    cam_origin,
                    ftp_stats=torch.tensor(
                        [[0, 0]],
                        dtype=torch.float32,
                        device=hf_seg.device,
                    ),
                    z=car_z,
                    deterministic=True,
                )
                color = get_img_without_pad(
                    color, sx, ex, sy, ey, psx, pex, psy, pey, img_cfg["PADDING"]
                )
                sigma = get_img_without_pad(
                    sigma[None, ...],
                    sx,
                    ex,
                    sy,
                    ey,
                    psx,
                    pex,
                    psy,
                    pey,
                    img_cfg["PADDING"],
                )
                _mask = sigma * g_mask[None, :, sy:ey, sx:ex]
                _mask[_mask < 0.1] = 0
                _mask = (_mask * 2).clamp(0, 1)
                car_img[:, :, sy:ey, sx:ex] = color * _mask
                car_mask[:, :, sy:ey, sx:ex] = _mask

    return car_img, car_mask


def _get_rotated_coordinates(point, pivot, theta):
    x = (
        (point["x"] - pivot["x"]) * math.cos(math.radians(theta))
        + (point["y"] - pivot["y"]) * math.sin(math.radians(theta))
        + pivot["x"]
    )
    y = (
        -(point["x"] - pivot["x"]) * math.sin(math.radians(theta))
        + (point["y"] - pivot["y"]) * math.cos(math.radians(theta))
        + pivot["y"]
    )

    new_pt = {"x": x, "y": y}
    if "z" in point:
        new_pt["z"] = point["z"]

    return new_pt


def _get_car_volume(car_inst, n_car_classes, vehicle_bev_file, vol_size, device):
    # In CitySample, all height values are delimetered by 2560.
    HF_DELIMETER = 2560

    car_sub_class = car_inst % n_car_classes + 1
    with open(vehicle_bev_file, "rb") as f:
        vehicles = pickle.load(f)
        projection = vehicles[car_sub_class - 1]
        projection["INS_BEV"] = projection["INS_BEV"].astype(np.int16) * car_sub_class
        projection = _get_padded_projection(projection, vol_size)

    seg_volume = torch.zeros(
        (vol_size, vol_size, vol_size),
        dtype=torch.int16,
        device=device,
    )
    seg_volume = extensions.footprint_extruder.extrude_footprint(
        seg_volume,
        torch.from_numpy(projection["INS_BEV"]).to(device).contiguous(),
        torch.from_numpy(projection["TD_HF"]).to(device).contiguous(),
        torch.from_numpy(projection["BU_HF"]).to(device).contiguous(),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    seg_volume[seg_volume != 0] = car_sub_class
    hf_seg, _ = get_hf_seg_tensor(
        projection,
        cx=None,
        cy=None,
        n_classes=n_car_classes + 1,
        cfg={"MAX_HEIGHT": HF_DELIMETER},
        output_device=device,
    )
    return hf_seg, seg_volume


def _get_padded_projection(projection, target_size):
    padded_projection = {}
    for k, v in projection.items():
        padded_projection[k] = np.zeros((target_size, target_size), dtype=v.dtype)
        sy = target_size // 2 - v.shape[0] // 2
        ey = sy + v.shape[0]
        sx = target_size // 2 - v.shape[1] // 2
        ex = sx + v.shape[1]
        padded_projection[k][sy:ey, sx:ex] = v
        # In the traffic scenario, the 0 degree is the east direction.
        # In NeRF, the 0 degree is the north direction.
        padded_projection[k] = np.ascontiguousarray(np.rot90(padded_projection[k]))

    return padded_projection


def get_video(frames, output_file, img_cfg):
    video = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"avc1"),
        4,
        (img_cfg["WIDTH"], img_cfg["HEIGHT"]),
    )
    for f in frames:
        video.write(f)

    video.release()


def main(
    patch_size,
    dataset,
    bg_ckpt,
    bldg_ckpt,
    car_ckpt,
    dataset_dirs,
    vehicle_bev_file,
    output_file,
):
    logging.info("Initialize models ...")
    bg_model, bldg_model, car_model = get_models(dataset, bg_ckpt, bldg_ckpt, car_ckpt)
    # Generate height fields and seg maps
    logging.info("Generating city layouts ...")
    projections, bldg_stats = get_city_layout(
        dataset,
        dataset_dirs,
        get_cfg_value("CLASSES", dataset)["BLDG_FACADE"],
        get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
        get_cfg_value("BLDG_INST_RANGE", dataset),
        get_cfg_value("BLDG_MAX_HEIGHT", dataset),
    )
    assert projections["REST"]["TD_HF"].shape == projections["REST"]["INS_BEV"].shape
    logging.info(
        "City Layout Patch Size (HxW): %s" % (projections["REST"]["TD_HF"].shape,)
    )

    # Generate camera trajectories
    logging.info("Generating camera poses ...")
    radius = (
        np.random.randint(128, 512)
        if dataset == "GOOGLE_EARTH"
        else np.random.randint(1024, 2048)
    )
    altitude = (
        np.random.randint(256, 512)
        if dataset == "GOOGLE_EARTH"
        else np.random.randint(1024, 2048)
    )
    logging.info("Radius = %d, Altitude = %s" % (radius, altitude))
    cam_poses = get_orbit_camera_positions(
        radius,
        altitude,
        get_cfg_value("VOL_SIZE/LAYOUT", dataset),
        get_cfg_value("N_VIEWPOINTS", dataset),
    )

    # Load the traffic scenarios
    scenarios = []
    if car_model is not None:
        logging.info("Loading traffic scenarios ...")
        scenarios = get_traffic_scenarios(dataset, dataset_dirs)
        assert len(cam_poses) <= len(scenarios)

    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z, bldg_zs, car_zs = get_latent_codes(
        bldg_stats,
        scenarios,
        bg_model.module.cfg.STYLE_DIM,
        bldg_model.output_device,
    )

    # Check the instance ID for buildings and cars
    assert min(list(bldg_zs.keys())) >= get_cfg_value("BLDG_INST_RANGE", dataset)[0]
    assert max(list(bldg_zs.keys())) < get_cfg_value("BLDG_INST_RANGE", dataset)[1]
    if car_zs is not None:
        assert min(list(car_zs.keys())) >= get_cfg_value("CAR_INST_RANGE", dataset)[0]
        assert max(list(car_zs.keys())) < get_cfg_value("CAR_INST_RANGE", dataset)[1]

    # Simply use image center as the patch center
    cy = projections["REST"]["TD_HF"].shape[0] // 2
    cx = projections["REST"]["TD_HF"].shape[1] // 2
    # Generate the concatenated height field and seg. map tensor
    hf_seg_static, static_inst = get_hf_seg_tensor(
        projections["REST"],
        cx,
        cy,
        get_cfg_value("N_LAYOUT_CLASSES", dataset),
        {
            "FACADE_CID": get_cfg_value("CLASSES", dataset)["BLDG_FACADE"],
            "INST_RANGE": get_cfg_value("BLDG_INST_RANGE", dataset),
            "MAX_HEIGHT": get_cfg_value("BLDG_MAX_HEIGHT", dataset),
            "VOL_SIZE": get_cfg_value("VOL_SIZE/EXTEND", dataset),
            "CLASS_MAPPER": _bldg_class_callback,
        },
        bg_model.output_device,
    )
    # Recalculate the instance positions based on the current patch
    bldg_stats = get_part_bldg_stats(
        static_inst,
        bldg_stats,
        cx,
        cy,
        get_cfg_value("BLDG_INST_RANGE", dataset),
    )

    # Build seg_volume
    logging.info("Generating seg volume ...")
    VOL_SIZES = {
        "LAYOUT": get_cfg_value("VOL_SIZE/LAYOUT", dataset),
        "BLDG": get_cfg_value("VOL_SIZE/BLDG", dataset),
        "CAR": get_cfg_value("VOL_SIZE/CAR", dataset),
        "EXT": get_cfg_value("VOL_SIZE/EXTEND", dataset),
    }
    IMG_CFG = {
        "HEIGHT": get_cfg_value("IMAGE_HEIGHT", dataset),
        "WIDTH": get_cfg_value("IMAGE_WIDTH", dataset),
        "PADDING": get_cfg_value("IMAGE_PADDING", dataset),
        "VFOV": get_cfg_value("IMAGE_VFOV", dataset),
    }

    logging.info("Caching raycast results ...")
    os.makedirs(get_cfg_value("RAYCAST_CACHE_DIR"), exist_ok=True)
    for f_idx, cp in enumerate(tqdm(cam_poses)):
        scenario = scenarios[f_idx] if scenarios else None
        traffic_projections = None
        if scenario:
            traffic_projections = (
                scripts.traffic_scenario_generator.get_traffic_bev_map(
                    scenario["METADATA"],
                    scenario["VEH_BEVS"],
                    projections["REST"]["INS_BEV"].shape,
                )
            )

        seg_volume = get_seg_volume(
            projections,
            traffic_projections,
            cx,
            cy,
            VOL_SIZES,
            {
                "ROOF_HEIGHT": get_cfg_value("BLDG_ROOF_HEIGHT", dataset),
                "ROOF_OFFSET": get_cfg_value("BLDG_ROOF_OFFSET", dataset),
                "INST_RANGE": get_cfg_value("BLDG_INST_RANGE", dataset),
                "MAX_HEIGHT": get_cfg_value("BLDG_MAX_HEIGHT", dataset),
            },
        )
        voxel_id, depth2, raydirs, cam_origin = get_voxel_intersection_perspective(
            seg_volume,
            cp,
            IMG_CFG,
            get_cfg_value("IMAGE_VFOV", dataset),
            get_cfg_value("N_VOXEL_SAMPLES", dataset),
        )
        with open(
            os.path.join(get_cfg_value("RAYCAST_CACHE_DIR"), "%04d.pkl" % f_idx),
            "wb",
        ) as fp:
            pickle.dump(
                (voxel_id, depth2, raydirs, cam_origin),
                fp,
            )
        # Remove seg_volume to save memory
        del voxel_id, depth2, raydirs, cam_origin, seg_volume
        torch.cuda.empty_cache()

    logging.info("Rendering videos ...")
    frames = []
    for f_idx, cp in enumerate(tqdm(cam_poses)):
        with open(
            os.path.join(get_cfg_value("RAYCAST_CACHE_DIR"), "%04d.pkl" % f_idx),
            "rb",
        ) as fp:
            voxel_id, depth2, raydirs, cam_origin = pickle.load(fp)

        scenario = copy.deepcopy(scenarios[f_idx]) if scenarios else None
        if scenario is not None:
            car_stats = get_part_car_stats(
                torch.unique(voxel_id).cpu().numpy(),
                scenario["METADATA"],
                cx,
                cy,
                get_cfg_value("CAR_INST_RANGE", dataset),
            )

        img = render(
            patch_size,
            {
                "STATIC": hf_seg_static,
            },
            {
                "VOXEL_ID": voxel_id,
                "DEPTH2": depth2,
                "RAYDIRS": raydirs,
                "CAM_ORIGIN": cam_origin,
            },
            {
                "BG": bg_model,
                "BLDG": bldg_model,
                "CAR": car_model if scenario else None,
            },
            {
                "BLDG": bldg_stats,
                "CAR": car_stats if scenario else None,
            },
            {
                "BG": bg_z,
                "BLDG": bldg_zs,
                "CAR": car_zs if scenario else None,
            },
            VOL_SIZES,
            IMG_CFG,
            {
                "BLDG_INST_RANGE": get_cfg_value("BLDG_INST_RANGE", dataset),
                "BLDG_INST_MULTIPLIER": get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
                "ROAD_CID": get_cfg_value("CLASSES", dataset)["ROAD"],
                "FACADE_CID": get_cfg_value("CLASSES", dataset)["BLDG_FACADE"]
                if dataset == "GOOGLE_EARTH"
                else 1,
                "ROOF_CID": get_cfg_value("CLASSES", dataset)["BLDG_ROOF"]
                if dataset == "GOOGLE_EARTH"
                else 2,
                "ROOF_OFFSET": get_cfg_value("BLDG_ROOF_OFFSET", dataset),
                "CAR_INST_RANGE": get_cfg_value("CAR_INST_RANGE", dataset),
                "N_CAR_CLASSES": get_cfg_value("N_CAR_CLASSES", dataset) - 1,
                "N_VOXEL_SAMPLES": get_cfg_value("N_VOXEL_SAMPLES", dataset),
                "CAR_BEV_FILE": vehicle_bev_file,
            },
        )
        img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
        frames.append(img[..., ::-1])
        cv2.imwrite("output/frames/%04d.jpg" % f_idx, img[..., ::-1])

    get_video(frames, output_file, IMG_CFG)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="GOOGLE_EARTH",
    )
    parser.add_argument(
        "--bg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "bg.pth"),
    )
    parser.add_argument(
        "--bldg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "bldg.pth"),
    )
    parser.add_argument(
        "--car_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "car.pth"),
    )
    parser.add_argument(
        "--city_osm_dir",
        default=os.path.join(PROJECT_HOME, "data", "osm", "US-NewYork"),
    )
    parser.add_argument(
        "--city_sample_dir",
        default=os.path.join(PROJECT_HOME, "data", "city-sample", "City02"),
    )
    parser.add_argument(
        "--vehicle_bev_file",
        default=os.path.join(PROJECT_HOME, "data", "city-sample", "vehicles.pkl"),
    )
    parser.add_argument(
        "--patch_height",
        default=get_cfg_value("IMAGE_HEIGHT") // 4,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=get_cfg_value("IMAGE_WIDTH") // 4,
        type=int,
    )
    parser.add_argument(
        "--output_file",
        default=os.path.join(PROJECT_HOME, "output", "rendering.mp4"),
        type=str,
    )
    args = parser.parse_args()

    main(
        (args.patch_height, args.patch_width),
        args.dataset,
        args.bg_ckpt,
        args.bldg_ckpt,
        args.car_ckpt,
        {"osm": args.city_osm_dir, "city_sample": args.city_sample_dir},
        args.vehicle_bev_file,
        args.output_file,
    )
