# -*- coding: utf-8 -*-
#
# @File:   inference.py
# @Author: Haozhe Xie
# @Date:   2023-05-31 15:01:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-07-15 21:08:09
# @Email:  root@haozhexie.com

import argparse
import copy
import cv2
import logging
import math
import numpy as np
import os
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
import utils.datasets
import utils.helpers


def _get_cfg_value(key, dataset=None):
    assert dataset in ["GOOGLE_EARTH", "CITY_SAMPLE", None]
    CONSTANTS = {
        "IMAGE_HEIGHT": 540,
        "IMAGE_WIDTH": 960,
        "IMAGE_PADDING": 8,
        "BLDG_ROOF_HEIGHT": 1,
        "N_VOXEL_SAMPLES": 6,
        "N_VIEWPOINTS": 24,
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
        "BLDG_INST_RANGE": "BLDG.INS_RANGE",
        "VOL_SIZE/LAYOUT": "VOL_SIZE",
        "VOL_SIZE/BLDG": "BLDG.VOL_SIZE",
        "VOL_SIZE/CAR": "CAR.VOL_SIZE",
    }
    # The constants are not defined in config.py but aligned with dataset_generator.py
    CFG_VALUES = {
        "IMAGE_VFOV": {"GOOGLE_EARTH": 36.86178122935623, "CITY_SAMPLE": None},
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

    ckpt = torch.load(ckpt_file_path)
    dt_cfg, model_cfg, inst = _get_compatible_cfg(ckpt["cfg"], dataset)
    city_dataset = utils.datasets.CityDataset(dt_cfg, None, inst)
    model = models.gancraft.GanCraftGenerator(
        cfg=model_cfg,
        n_classes=city_dataset.get_n_classes(),
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
        car_ckpt = _get_model("%s_CAR" % dataset, car_ckpt)

    return bg_model, bldg_model, car_model


def get_osm_city_layout(
    city_osm_dir, bldg_facade_cid, bldg_inst_mult, min_bldg_inst, bldg_max_height
):
    hf = np.array(Image.open(os.path.join(city_osm_dir, "hf.png")))
    seg = np.array(Image.open(os.path.join(city_osm_dir, "seg.png")).convert("P"))
    ins_seg, building_stats = _get_instance_seg_layout(
        seg, bldg_facade_cid, bldg_inst_mult, min_bldg_inst
    )
    hf = _clip_height_field(hf, bldg_max_height)

    return hf.astype(np.int32), ins_seg.astype(np.int32), building_stats


def _get_instance_seg_layout(
    seg_layout, bldg_facade_cid, bldg_inst_mult, min_bldg_inst
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
    labels = (labels + min_bldg_inst) * bldg_inst_mult

    seg_layout[seg_layout == bldg_facade_cid] = 0
    seg_layout = seg_layout * (1 - building_mask) + labels * building_mask
    assert np.max(labels) < 2147483648
    return seg_layout.astype(np.int32), stats[:, :4]


def _clip_height_field(hf, layout_max_height):
    hf[hf >= layout_max_height] = layout_max_height - 1
    return hf


def get_latent_codes(
    bldg_stats, bg_style_dim, bldg_inst_mult, min_bldg_inst, output_device
):
    bg_z = _get_z(output_device, bg_style_dim)
    building_zs = {
        (i + min_bldg_inst) * bldg_inst_mult: _get_z(output_device)
        for i in range(len(bldg_stats))
    }
    return bg_z, building_zs


def _get_z(device, z_dim=256):
    if z_dim is None:
        return None

    return torch.randn(1, z_dim, dtype=torch.float32, device=device)


def get_image_patch(image, cx, cy, patch_size):
    sx = cx - patch_size // 2
    sy = cy - patch_size // 2
    ex = sx + patch_size
    ey = sy + patch_size
    return image[sy:ey, sx:ex]


def get_part_hf_seg(hf, seg, cx, cy, patch_size):
    part_hf = get_image_patch(hf, cx, cy, patch_size)
    part_seg = get_image_patch(seg, cx, cy, patch_size)
    assert part_hf.shape == (
        patch_size,
        patch_size,
    ), part_hf.shape
    assert part_hf.shape == part_seg.shape, part_seg.shape
    return part_hf, part_seg


def get_part_building_stats(
    part_seg, bldg_stats, cx, cy, bldg_inst_mult, min_bldg_inst
):
    _buildings = np.unique(part_seg[part_seg > min_bldg_inst])
    _bldg_stats = {}
    for b in _buildings:
        _b = b // bldg_inst_mult - min_bldg_inst
        _bldg_stats[b] = [
            bldg_stats[_b, 1] - cy + bldg_stats[_b, 3] / 2,
            bldg_stats[_b, 0] - cx + bldg_stats[_b, 2] / 2,
        ]
    return _bldg_stats


def get_hf_seg_tensor(
    part_hf, part_seg, bldg_max_height, n_layout_classes, output_device
):
    part_hf = torch.from_numpy(part_hf[None, None, ...]).to(output_device)
    part_seg = torch.from_numpy(part_seg[None, None, ...]).to(output_device)
    part_hf = part_hf / bldg_max_height
    part_seg = utils.helpers.masks_to_onehots(part_seg[:, 0, :, :], n_layout_classes)
    return torch.cat([part_hf, part_seg], dim=1)


def get_seg_volume(part_hf, part_seg, vol_sizes, bldg_cfg):
    if part_hf.shape == (vol_sizes["EXT"], vol_sizes["EXT"]):
        part_hf = part_hf[
            vol_sizes["BLDG"] : -vol_sizes["BLDG"],
            vol_sizes["BLDG"] : -vol_sizes["BLDG"],
        ]
        part_seg = part_seg[
            vol_sizes["BLDG"] : -vol_sizes["BLDG"],
            vol_sizes["BLDG"] : -vol_sizes["BLDG"],
        ]

    assert part_hf.shape == (vol_sizes["LAYOUT"], vol_sizes["LAYOUT"])
    assert part_hf.shape == part_seg.shape, part_seg.shape
    footprint_extruder = extensions.footprint_extruder.FootprintExtruder(
        roof_height=bldg_cfg["ROOF_HEIGHT"],
        roof_id_offset=bldg_cfg["ROOF_OFFSET"],
        footprint_id_range=bldg_cfg["INST_RANGE"],
        max_height=bldg_cfg["MAX_HEIGHT"],
    )

    seg_volume = footprint_extruder(
        torch.from_numpy(part_hf[None, None, ...]).cuda(),
        torch.from_numpy(part_seg[None, None, ...]).cuda(),
    ).squeeze()
    logging.debug("The shape of SegVolume: %s" % (seg_volume.size(),))

    # Change the top-level voxel of the "Building Facade" to "Building Roof"
    roof_seg_map = part_seg.copy()
    non_roof_msk = part_seg <= bldg_cfg["INST_RANGE"][0]
    # Assume the ID of a facade instance is 2k (4k), the corresponding roof instance is 2k-1 (4k+1).
    roof_seg_map = roof_seg_map + bldg_cfg["ROOF_OFFSET"]
    roof_seg_map[non_roof_msk] = 0
    for rh in range(1, bldg_cfg["ROOF_HEIGHT"] + 1):
        seg_volume = seg_volume.scatter_(
            dim=2,
            index=torch.from_numpy(part_hf[..., None] + rh).long().cuda(),
            src=torch.from_numpy(roof_seg_map[..., None]).cuda(),
        )
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    return seg_volume


def get_orbit_camera_positions(radius, altitude, vol_size_layout, n_viewpoints):
    # TODO: More flexible for CITY_SAMPLE
    camera_positions = []
    cx = vol_size_layout // 2
    cy = cx
    for i in range(n_viewpoints):
        theta = 2 * math.pi / n_viewpoints * i
        cam_x = cx + radius * math.cos(theta)
        cam_y = cy + radius * math.sin(theta)
        camera_positions.append({"x": cam_x, "y": cam_y, "z": altitude})

    return camera_positions


def get_voxel_intersection_perspective(
    seg_volume, camera_location, img_sizes, img_vfov, n_voxel_samples
):
    CAMERA_FOCAL = img_sizes["HEIGHT"] / 2 / np.tan(np.deg2rad(img_vfov)) * 2.06
    # print(seg_volume.size())  # torch.Size([1536, 1536, 640])
    camera_target = {
        "x": seg_volume.size(1) // 2 - 1,
        "y": seg_volume.size(0) // 2 - 1,
    }
    cam_origin = torch.tensor(
        [
            camera_location["y"],
            camera_location["x"],
            camera_location["z"],
        ],
        dtype=torch.float32,
        device=seg_volume.device,
    )

    voxel_id, depth2, raydirs = extensions.voxlib.ray_voxel_intersection_perspective(
        seg_volume,
        cam_origin,
        torch.tensor(
            [
                camera_target["y"] - camera_location["y"],
                camera_target["x"] - camera_location["x"],
                -camera_location["z"],
            ],
            dtype=torch.float32,
            device=seg_volume.device,
        ),
        torch.tensor([0, 0, 1], dtype=torch.float32),
        CAMERA_FOCAL,
        [
            (img_sizes["HEIGHT"] - 1) / 2.0,
            (img_sizes["WIDTH"] - 1) / 2.0,
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
    bldg_cfg,
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
    _voxel_id[
        (voxel_id >= bldg_cfg["INST_RANGE"][0]) & (voxel_id < bldg_cfg["INST_RANGE"][1])
    ] = bldg_cfg["FACADE_CID"]
    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()
    bg_img = torch.zeros(
        1,
        3,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=bg_model.output_device,
    )
    # Render background patches by patch to avoid OOM
    for i in range(img_cfg["HEIGHT"] // patch_size[0]):
        for j in range(img_cfg["WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(
                sx, ex, sy, ey, img_cfg["HEIGHT"], img_cfg["WIDTH"], img_cfg["PADDING"]
            )
            output_bg = bg_model(
                hf_seg=hf_seg,
                voxel_id=_voxel_id[:, psy:pey, psx:pex],
                depth2=depth2[:, psy:pey, psx:pex],
                raydirs=raydirs[:, psy:pey, psx:pex],
                cam_origin=cam_origin,
                bldg_stats=None,
                z=z,
                deterministic=True,
            )
            # Make road blurry
            road_mask = (
                (_voxel_id[:, None, psy:pey, psx:pex, 0, 0] == bldg_cfg["ROAD_CID"])
                .repeat(1, 3, 1, 1)
                .float()
            )
            output_bg = blurrer(output_bg) * road_mask + output_bg * (1 - road_mask)
            bg_img[:, :, sy:ey, sx:ex] = get_img_without_pad(
                output_bg, sx, ex, sy, ey, psx, pex, psy, pey, img_cfg["PADDING"]
            )

    return bg_img


def render_bldg(
    patch_size,
    gancraft_fg,
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
    bldg_cfg,
):
    _voxel_id = copy.deepcopy(voxel_id)
    _curr_bldg = torch.tensor(
        [curr_bldg_inst, curr_bldg_inst + bldg_cfg["ROOF_OFFSET"]],
        device=voxel_id.device,
    )
    _voxel_id[~torch.isin(_voxel_id, _curr_bldg)] = 0
    _voxel_id[voxel_id == curr_bldg_inst] = bldg_cfg["FACADE_CID"]
    _voxel_id[voxel_id == curr_bldg_inst - 1] = bldg_cfg["ROOF_CID"]
    # assert (_voxel_id < CONSTANTS["LAYOUT_N_CLASSES"]).all()

    _hf_seg = copy.deepcopy(hf_seg)
    _hf_seg[hf_seg != curr_bldg_inst] = 0
    _hf_seg[hf_seg == curr_bldg_inst] = bldg_cfg["FACADE_CID"]
    _raydirs = copy.deepcopy(raydirs)
    _raydirs[_voxel_id[..., 0, 0] == 0] = 0

    # Crop the "hf_seg" image using the center of the target building as the reference
    cx = vol_sizes["EXT"] // 2 - int(bldg_stats[1])
    cy = vol_sizes["EXT"] // 2 - int(bldg_stats[0])
    sx = cx - vol_sizes["BLDG"] // 2
    ex = cx + vol_sizes["BLDG"] // 2
    sy = cy - vol_sizes["BLDG"] // 2
    ey = cy + vol_sizes["BLDG"] // 2
    _hf_seg = hf_seg[:, :, sy:ey, sx:ex]

    fg_img = torch.zeros(
        1,
        3,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    fg_mask = torch.zeros(
        1,
        1,
        img_cfg["HEIGHT"],
        img_cfg["WIDTH"],
        dtype=torch.float32,
        device=gancraft_fg.output_device,
    )
    # Prevent some buildings are out of bound.
    # THIS SHOULD NEVER HAPPEN AGAIN.
    # if (
    #     _hf_seg.size(2) != vol_sizes["BLDG"]
    #     or _hf_seg.size(3) != vol_sizes["BLDG"]
    # ):
    #     return fg_img, fg_mask

    # Render foreground patches by patch to avoid OOM
    for i in range(img_cfg["HEIGHT"] // patch_size[0]):
        for j in range(img_cfg["WIDTH"] // patch_size[1]):
            sy, sx = i * patch_size[0], j * patch_size[1]
            ey, ex = sy + patch_size[0], sx + patch_size[1]
            psx, pex, psy, pey = get_pad_img_bbox(
                sx, ex, sy, ey, img_cfg["HEIGHT"], img_cfg["WIDTH"], img_cfg["PADDING"]
            )

            if torch.count_nonzero(_raydirs[:, sy:ey, sx:ex]) > 0:
                output_fg = gancraft_fg(
                    _hf_seg,
                    _voxel_id[:, psy:pey, psx:pex],
                    depth2[:, psy:pey, psx:pex],
                    _raydirs[:, psy:pey, psx:pex],
                    cam_origin,
                    bldg_stats=torch.from_numpy(np.array(bldg_stats)).unsqueeze(dim=0),
                    z=building_z,
                    deterministic=True,
                )
                facade_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0] == curr_bldg_inst
                ).unsqueeze(dim=1)
                roof_mask = (
                    voxel_id[:, sy:ey, sx:ex, 0, 0]
                    == curr_bldg_inst + bldg_cfg["ROOF_OFFSET"]
                ).unsqueeze(dim=1)
                facade_img = facade_mask * get_img_without_pad(
                    output_fg, sx, ex, sy, ey, psx, pex, psy, pey, img_cfg["PADDING"]
                )
                # Make roof blurry
                # output_fg = F.interpolate(
                #     F.interpolate(output_fg * 0.8, scale_factor=0.75),
                #     scale_factor=4 / 3,
                # ),
                roof_img = roof_mask * get_img_without_pad(
                    output_fg,
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
                fg_mask[:, :, sy:ey, sx:ex] = torch.logical_or(facade_mask, roof_mask)
                fg_img[:, :, sy:ey, sx:ex] = (
                    facade_img * facade_mask + roof_img * roof_mask
                )

    return fg_img, fg_mask


def render_static(
    patch_size,
    hf_seg,
    voxel_id,
    depth2,
    raydirs,
    cam_origin,
    bg_model,
    bldg_model,
    bldg_stats,
    bg_z,
    bldg_zs,
    vol_sizes,
    img_cfg,
    bldg_cfg,
):
    buildings = torch.unique(
        voxel_id[
            (voxel_id >= bldg_cfg["INST_RANGE"][0])
            & (voxel_id < bldg_cfg["INST_RANGE"][1])
        ]
    )
    buildings = buildings[buildings % bldg_cfg["MULTIPLIER"] == 0]
    with torch.no_grad():
        bg_img = render_bg(
            patch_size,
            bg_model,
            hf_seg,
            voxel_id,
            depth2,
            raydirs,
            cam_origin,
            bg_z,
            vol_sizes,
            img_cfg,
            bldg_cfg,
        )
        for b in buildings:
            fg_img, fg_mask = render_bldg(
                patch_size,
                bldg_model,
                b.item(),
                hf_seg,
                voxel_id,
                depth2,
                raydirs,
                cam_origin,
                bldg_stats[b.item()],
                bldg_zs[b.item()],
                vol_sizes,
                img_cfg,
                bldg_cfg,
            )
            bg_img = bg_img * (1 - fg_mask) + fg_img * fg_mask

    return bg_img


def get_video(frames, output_file, img_height, img_width):
    video = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"avc1"), 4, (img_width, img_height)
    )
    for f in frames:
        video.write(f)

    video.release()


def main(patch_size, dataset, bg_ckpt, bldg_ckpt, car_ckpt, city_osm_dir, output_file):
    # TODO: car_model
    logging.info("Initialize models ...")
    bg_model, bldg_model, car_model = get_models(dataset, bg_ckpt, bldg_ckpt, car_ckpt)
    # Generate height fields and seg maps
    logging.info("Generating city layouts ...")
    hf, seg, bldg_stats = get_osm_city_layout(
        city_osm_dir,
        _get_cfg_value("CLASSES", dataset)["BLDG_FACADE"],
        _get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
        _get_cfg_value("BLDG_INST_RANGE", dataset)[0],
        _get_cfg_value("BLDG_MAX_HEIGHT", dataset),
    )
    assert hf.shape == seg.shape
    logging.info("City Layout Patch Size (HxW): %s" % (hf.shape,))

    # Generate latent codes
    logging.info("Generating latent codes ...")
    bg_z, bldg_zs = get_latent_codes(
        bldg_stats,
        bg_model.module.cfg.STYLE_DIM,
        _get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
        _get_cfg_value("BLDG_INST_RANGE", dataset)[0],
        bldg_model.output_device,
    )

    # Simply use image center as the patch center
    cy, cx = seg.shape[0] // 2, seg.shape[1] // 2
    # Generate local image patch of the height field and seg map
    part_hf, part_seg = get_part_hf_seg(
        hf, seg, cx, cy, _get_cfg_value("VOL_SIZE/EXTEND", dataset)
    )
    # print(part_hf.shape)    # (2880, 2880)
    # print(part_seg.shape)   # (2880, 2880)

    # Recalculate the building positions based on the current patch
    bldg_stats = get_part_building_stats(
        part_seg,
        bldg_stats,
        cx,
        cy,
        _get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
        _get_cfg_value("BLDG_INST_RANGE", dataset)[0],
    )
    # Generate the concatenated height field and seg. map tensor
    hf_seg = get_hf_seg_tensor(
        part_hf,
        part_seg,
        _get_cfg_value("BLDG_MAX_HEIGHT", dataset),
        _get_cfg_value("N_LAYOUT_CLASSES", dataset),
        bg_model.output_device,
    )
    # print(hf_seg.size())    # torch.Size([1, 8, 2880, 2880])
    # Build seg_volume
    logging.info("Generating seg volume ...")
    VOL_SIZES = {
        "LAYOUT": _get_cfg_value("VOL_SIZE/LAYOUT", dataset),
        "BLDG": _get_cfg_value("VOL_SIZE/BLDG", dataset),
        "EXT": _get_cfg_value("VOL_SIZE/EXTEND", dataset),
    }
    seg_volume = get_seg_volume(
        part_hf,
        part_seg,
        VOL_SIZES,
        {
            "ROOF_HEIGHT": _get_cfg_value("BLDG_ROOF_HEIGHT", dataset),
            "ROOF_OFFSET": _get_cfg_value("BLDG_ROOF_OFFSET", dataset),
            "INST_RANGE": _get_cfg_value("BLDG_INST_RANGE", dataset),
            "MAX_HEIGHT": _get_cfg_value("BLDG_MAX_HEIGHT", dataset),
        },
    )

    # Generate camera trajectories
    logging.info("Generating camera poses ...")
    radius = np.random.randint(128, 512)
    altitude = np.random.randint(256, 512)
    logging.info("Radius = %d, Altitude = %s" % (radius, altitude))
    cam_pos = get_orbit_camera_positions(
        radius,
        altitude,
        _get_cfg_value("VOL_SIZE/LAYOUT", dataset),
        _get_cfg_value("N_VIEWPOINTS", dataset),
    )

    logging.info("Rendering videos ...")
    IMG_CFG = {
        "HEIGHT": _get_cfg_value("IMAGE_HEIGHT", dataset),
        "WIDTH": _get_cfg_value("IMAGE_WIDTH", dataset),
        "PADDING": _get_cfg_value("IMAGE_PADDING", dataset),
    }
    frames = []
    for _, cp in enumerate(tqdm(cam_pos)):
        voxel_id, depth2, raydirs, cam_origin = get_voxel_intersection_perspective(
            seg_volume,
            cp,
            IMG_CFG,
            _get_cfg_value("IMAGE_VFOV", dataset),
            _get_cfg_value("N_VOXEL_SAMPLES", dataset),
        )
        img = render_static(
            patch_size,
            hf_seg,
            voxel_id,
            depth2,
            raydirs,
            cam_origin,
            bg_model,
            bldg_model,
            bldg_stats,
            bg_z,
            bldg_zs,
            VOL_SIZES,
            IMG_CFG,
            {
                "INST_RANGE": _get_cfg_value("BLDG_INST_RANGE", dataset),
                "MULTIPLIER": _get_cfg_value("BLDG_INST_MULTIPLIER", dataset),
                "ROAD_CID": _get_cfg_value("CLASSES", dataset)["ROAD"],
                "FACADE_CID": _get_cfg_value("CLASSES", dataset)["BLDG_FACADE"],
                "ROOF_CID": _get_cfg_value("CLASSES", dataset)["BLDG_ROOF"],
                "ROOF_OFFSET": _get_cfg_value("BLDG_ROOF_OFFSET", dataset),
            },
        )
        img = (utils.helpers.tensor_to_image(img, "RGB") * 255).astype(np.uint8)
        frames.append(img[..., ::-1])
        cv2.imwrite("output/test.jpg", img[..., ::-1])

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
        default=os.path.join(PROJECT_HOME, "output", "gancraft-bg.pth"),
    )
    parser.add_argument(
        "--bldg_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-bldg.pth"),
    )
    parser.add_argument(
        "--car_ckpt",
        default=os.path.join(PROJECT_HOME, "output", "gancraft-car.pth"),
    )
    parser.add_argument(
        "--city_osm_dir",
        default=os.path.join(PROJECT_HOME, "data", "osm", "US-NewYork"),
    )
    parser.add_argument(
        "--patch_height",
        default=_get_cfg_value("IMAGE_HEIGHT") // 4,
        type=int,
    )
    parser.add_argument(
        "--patch_width",
        default=_get_cfg_value("IMAGE_WIDTH") // 4,
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
        args.city_osm_dir,
        args.output_file,
    )
