# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-08-13 09:22:28
# @Email:  root@haozhexie.com

import argparse
import cv2
import csv
import json
import logging
import logging.config
import numpy as np
import os
import pickle
import scipy
import sys
import torch

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.footprint_extruder
import extensions.voxlib
import utils.helpers


def get_cfg_value(key):
    CONSTANTS = {
        "IMAGE_HEIGHT": 540,
        "IMAGE_WIDTH": 960,
    }

    if key in CONSTANTS:
        return CONSTANTS[key]
    else:
        return _get_dataset_cfg_value(key)


def _get_dataset_cfg_value(key):
    from config import cfg

    CFG_KEYS = {
        "MAX_HEIGHT": "MAX_HEIGHT",
        "BLDG_INST_RANGE": "BLDG.INS_RANGE",
        "CAR_INST_RANGE": "CAR.INS_RANGE",
    }
    CFG_VALUES = {
        "SCALE": 5,
        "Z_OFFSET": 17.5,  # 17.5 = -3.5m * scale -> 0 for the water plane
        "VOL_SIZE": 3072,
        "BEV_MAP_SIZE": 24576,
        "BLDG_ROOF_HEIGHT": 1,
        "BLDG_ROOF_OFFSET": 1,
    }
    CLASSES = {
        "NULL": 0,
        "ROAD": 1,
        "FWY_DECK": 1,
        "FWY_PILLAR": 2,
        "FWY_BARRIER": 2,
        "CAR": 3,
        "WATER": 4,
        "SKY": 5,
        "ZONE": 6,
        "BLDG_FACADE": 7,
        "BLDG_ROOF": 8,
    }
    if key == "CLASSES":
        return CLASSES
    elif key in CFG_KEYS:
        # Read the value from the dataset config recursively
        _cfg = cfg.DATASETS.CITY_SAMPLE
        path = CFG_KEYS[key].split(".")
        for p in path:
            if p not in _cfg:
                return None
            _cfg = getattr(_cfg, p)

        return _cfg
    elif key in CFG_VALUES:
        return CFG_VALUES[key]
    else:
        return None


def get_projections(city_dir, map_size, z_offset, classes, inst_ranges):
    # The constants defined in HOU_CLASSES only used in this function.
    HOU_CLASSES = {
        "ROAD": 1,
        "FWY_DECK": 2,
        "FWY_PILLAR": 3,
        "FWY_BARRIER": 4,
        "ZONE": 5,
        # The following classes are not appeared in the Houdini export
        # "NULL": 0,
        # "BLDG_FACADE": 6,
        # "BLDG_ROOF": 7,
    }
    HOU_INV_INDEX = {v: k for k, v in HOU_CLASSES.items()}
    HOU_SCALES = {
        "ROAD": 10,
        "FWY_DECK": 10,
        "FWY_PILLAR": 5,
        "FWY_BARRIER": 8,
        "CAR": 2,
        "WATER": 50,
        "ZONE": 10,
        "BLDG_FACADE": 5,
    }

    points_file_path = os.path.join(city_dir, "Points.pkl")
    if not os.path.exists(points_file_path):
        logging.warning("File not found in %s" % (points_file_path))
        return {}

    with open(points_file_path, "rb") as fp:
        points = pickle.load(fp)
    # Make all the point coordinates positive at z-axis
    points[:, 2] += z_offset

    # Separate the points into three categories: CAR, FWY, and REST
    car_rows = (points[:, 3] >= inst_ranges["CAR"][0]) & (
        points[:, 3] < inst_ranges["CAR"][1]
    )
    fwy_rows = np.isin(
        points[:, 3],
        [
            HOU_CLASSES["FWY_DECK"],
            HOU_CLASSES["FWY_PILLAR"],
            HOU_CLASSES["FWY_BARRIER"],
        ],
    )
    rest_rows = ~np.logical_or(car_rows, fwy_rows)

    projections = {
        "CAR": _get_projection(
            points[car_rows], map_size, HOU_INV_INDEX, classes, HOU_SCALES, inst_ranges
        ),
        "FREEWAY": _get_projection(
            points[fwy_rows], map_size, HOU_INV_INDEX, classes, HOU_SCALES, inst_ranges
        ),
        "REST": _get_projection(
            points[rest_rows], map_size, HOU_INV_INDEX, classes, HOU_SCALES, inst_ranges
        ),
    }
    projections["REST"] = _get_water_areas(projections["REST"], classes)
    return projections


def _get_projection(points, map_size, hou_inv_idx, classes, scales, inst_ranges):
    # assert points.dtype == np.int16
    ins_map = np.zeros((map_size, map_size), dtype=points.dtype)
    tpd_hf = -1 * np.ones((map_size, map_size), dtype=points.dtype)
    btu_hf = np.iinfo(points.dtype).max * np.ones(
        (map_size, map_size), dtype=points.dtype
    )
    for p in tqdm(points, leave=False):
        x, y, z, inst = p
        if z < 0:
            continue

        c_name = hou_inv_idx[inst] if inst in hou_inv_idx else None
        if c_name is None:
            if inst >= inst_ranges["BLDG"][0] and inst < inst_ranges["BLDG"][1]:
                # No building roof instance ID in the Houdini export.
                assert inst % 4 == 0
                c_name = "BLDG_FACADE"
            elif inst >= inst_ranges["CAR"][0] and inst < inst_ranges["CAR"][1]:
                c_name = "CAR"
            else:
                raise ValueError("Unknown instance ID: %d" % inst)

        s = scales[c_name]
        x += map_size // 2
        y += map_size // 2
        if tpd_hf[y, x] < z:
            tpd_hf[y : y + s, x : x + s] = z
            ins_map[y : y + s, x : x + s] = (
                classes[c_name]
                if c_name not in ["BLDG_FACADE", "BLDG_ROOF", "CAR"]
                else inst
            )
        if btu_hf[y, x] > z:
            btu_hf[y : y + s, x : x + s] = z

    return {
        "INS_BEV": ins_map,
        "TD_HF": tpd_hf,
        "BU_HF": btu_hf,
    }


def _get_water_areas(projection, classes):
    # The rest areas are assigned as the water areas
    water_area = projection["INS_BEV"] == classes["NULL"]
    projection["INS_BEV"][water_area] = classes["WATER"]
    # Set water plane height to 1 [MAGIC NUMBER]
    projection["TD_HF"][water_area] = 1
    projection["BU_HF"][water_area] = 0
    return projection


def get_instance_bboxes(seg_map, inst_range):
    building_instances = [
        i for i in np.unique(seg_map) if i >= inst_range[0] and i < inst_range[1]
    ]
    bboxes = {}
    for bi in tqdm(
        building_instances, desc="Generating Building Bounding Boxes", leave=False
    ):
        bboxes[bi] = cv2.boundingRect((seg_map == bi).astype(np.uint8))

    return bboxes


def get_camera_poses(cam_pose, half_map_size, scale, depth_offset):
    cam_pose["tx"] = float(cam_pose["tx"]) / 100 * scale + half_map_size
    cam_pose["ty"] = float(cam_pose["ty"]) / 100 * scale + half_map_size
    cam_pose["tz"] = float(cam_pose["tz"]) / 100 * scale + depth_offset
    cam_position = np.array([cam_pose["tx"], cam_pose["ty"], cam_pose["tz"]])
    cam_look_at = _get_look_at_position(
        cam_position,
        np.array(
            [
                float(cam_pose["qx"]),
                float(cam_pose["qy"]),
                float(cam_pose["qz"]),
                float(cam_pose["qw"]),
            ]
        ),
    )
    return {
        "cam_position": cam_position,
        "cam_look_at": cam_look_at,
    }


def _get_look_at_position(cam_position, cam_quaternion):
    mat3 = scipy.spatial.transform.Rotation.from_quat(cam_quaternion).as_matrix()
    step = cam_position[-1] / mat3[:3, 0][-1]  # Make z=0 for the look-at position
    return cam_position - step * mat3[:3, 0]


def get_bev_map_bbox(cam_pose, patch_size, scale=1):
    assert scale == 1, "Not implemented for scale != 1"
    scaled_patch_size = int(patch_size / scale)
    half_s_patch_size = scaled_patch_size // 2

    delta = cam_pose["cam_look_at"][:2] - cam_pose["cam_position"][:2]
    if (np.abs(delta) <= half_s_patch_size).all():
        patch_center = _get_max_square_center(
            cam_pose["cam_look_at"][:2], delta, half_s_patch_size
        )
    else:
        # Camera is too far away from the look_at point, crop from the camera center
        patch_center = _get_max_square_center(
            cam_pose["cam_position"][:2], delta, half_s_patch_size
        )
    # Ordered by: (x, y)
    patch_center = (patch_center + 0.5).astype(np.int32)
    top_left = patch_center - half_s_patch_size
    btm_right = patch_center + half_s_patch_size
    return {"TL": top_left, "BR": btm_right}


def _get_max_square_center(viewpoint, delta, half_size):
    dx_scale = half_size / np.abs(delta[0])
    dy_scale = half_size / np.abs(delta[1])
    # Use the smaller scale
    if dx_scale < dy_scale:
        cx = viewpoint[0] + np.sign(delta[0]) * half_size
        cy = viewpoint[1] + delta[1] * dx_scale
    else:
        cy = viewpoint[1] + np.sign(delta[1]) * half_size
        cx = viewpoint[0] + delta[0] * dy_scale

    return np.array([cx, cy])


def get_volume_with_scale(projections, bev_map_bbox, bldg_cfg, vol_size):
    fe = extensions.footprint_extruder.FootprintExtruder(
        roof_height=bldg_cfg["ROOF_HEIGHT"],
        roof_id_offset=bldg_cfg["ROOF_OFFSET"],
        bldg_inst_range=bldg_cfg["INST_RANGE"],
    )

    # TODO: Consider separating the volume into different classes (especially cars)
    volume = torch.zeros(
        (vol_size, vol_size, bldg_cfg["MAX_HEIGHT"]),
        dtype=torch.int16,
        device="cuda:0",
    )
    for k in ["CAR", "FREEWAY", "REST"]:
        _projections = _get_projection_patch(
            projections[k], bev_map_bbox, vol_size, volume.device
        )
        # TODO: Uncomment
        assert torch.min(_projections["TD_HF"]) >= 0
        assert torch.max(_projections["TD_HF"]) < bldg_cfg["MAX_HEIGHT"]

        volume = fe(
            volume,
            _projections["INS_BEV"],
            _projections["TD_HF"],
            _projections["BU_HF"],
        )

    return volume.squeeze(dim=0)


def _get_projection_patch(projections, bev_map_bbox, patch_size, device):
    INTERPOLATION = {
        "INS_BEV": cv2.INTER_NEAREST,
        "TD_HF": cv2.INTER_LINEAR,
        "BU_HF": cv2.INTER_LINEAR,
    }
    # Crop to patches
    patches = {}
    for k, v in INTERPOLATION.items():
        tl, br = bev_map_bbox["TL"], bev_map_bbox["BR"]
        _patch = projections[k][tl[1] : br[1], tl[0] : br[0]].astype(np.int16)
        if _patch.shape != (patch_size, patch_size):
            _patch = cv2.resize(_patch, (patch_size, patch_size), interpolation=v)

        patches[k] = utils.helpers.var_or_cuda(
            torch.from_numpy(_patch),
            device,
        )

    return patches


def get_ray_voxel_intersection(cam_rig, cam_pose, volume, classes):
    N_MAX_SAMPLES = 6
    cam_origin = torch.tensor(
        [
            cam_pose["cam_position"][1],
            cam_pose["cam_position"][0],
            cam_pose["cam_position"][2],
        ],
        dtype=torch.float32,
        device=volume.device,
    )
    viewdir = torch.tensor(
        [
            cam_pose["cam_look_at"][1] - cam_pose["cam_position"][1],
            cam_pose["cam_look_at"][0] - cam_pose["cam_position"][0],
            cam_pose["cam_look_at"][2] - cam_pose["cam_position"][2],
        ],
        dtype=torch.float32,
        device=volume.device,
    )
    (
        voxel_id,
        depth2,
        raydirs,
    ) = extensions.voxlib.ray_voxel_intersection_perspective(
        volume,
        cam_origin,
        viewdir,
        torch.tensor([0, 0, 1], dtype=torch.float32),
        cam_rig["intrinsics"][0],
        [
            cam_rig["sensor_size"][1] / 2,
            cam_rig["sensor_size"][0] / 2,
        ],
        [cam_rig["sensor_size"][1], cam_rig["sensor_size"][0]],
        N_MAX_SAMPLES,
    )
    # Bug Fix: Map NULL voxels to SKY
    # TODO: Uncomment
    # voxel_id[voxel_id == 0] = classes["SKY"]

    return {
        "voxel_id": voxel_id,
        "depth2": depth2,
        "raydirs": raydirs,
        "viewdir": viewdir,
        "cam_origin": cam_origin,
    }


# def get_ambiguous_seg_mask(voxel_id, est_seg_map):
#     ins_seg_map = voxel_id.squeeze()[..., 0].copy()
#     # NOTE: In ins_seg_map, 4n and 4n+1 denote building facade and roof, respectively.
#     #       In est_seg_map, 7 and 8 denote building facade and roof, respectively.
#     ins_seg_map[ins_seg_map >= CONSTANTS["CAR_INS_MIN_ID"]] = CONSTANTS["CAR_CLS_ID"]
#     ins_seg_map[
#         (ins_seg_map >= CONSTANTS["BLD_INS_MIN_ID"]) & (ins_seg_map % 4 == 0)
#     ] = CONSTANTS["BLD_FACADE_CLS_ID"]
#     ins_seg_map[
#         (ins_seg_map >= CONSTANTS["BLD_INS_MIN_ID"]) & (ins_seg_map % 4 == 1)
#     ] = CONSTANTS["BLD_ROOF_CLS_ID"]
#     return ins_seg_map == np.array(est_seg_map.convert("P"))


def main(data_dir, seg_map_file_pattern, img_size, is_debug):
    cities = sorted(os.listdir(data_dir))
    INST_RANGES = {
        "CAR": get_cfg_value("CAR_INST_RANGE"),
        "BLDG": get_cfg_value("BLDG_INST_RANGE"),
    }
    for city in tqdm(cities):
        city_dir = os.path.join(data_dir, city)
        proj_dir = os.path.join(city_dir, "Projections")
        if not os.path.exists(proj_dir):
            logging.info("Generating Projections for %s" % city)
            projections = get_projections(
                city_dir,
                get_cfg_value("BEV_MAP_SIZE"),
                get_cfg_value("Z_OFFSET"),
                get_cfg_value("CLASSES"),
                INST_RANGES,
            )
            os.makedirs(proj_dir, exist_ok=True)
            for k, v in projections.items():
                assert k in ["CAR", "FREEWAY", "REST"]
                for mk, mv in v.items():
                    assert mk in ["INS_BEV", "TD_HF", "BU_HF"]
                    Image.fromarray(mv).save(
                        os.path.join(proj_dir, "%s_%s.png" % (k, mk))
                    )
        else:
            logging.info("Reading projections for %s" % city)
            projections = {}
            for k in ["CAR", "FREEWAY", "REST"]:
                projections[k] = {
                    mk: np.array(
                        Image.open(os.path.join(proj_dir, "%s_%s.png" % (k, mk)))
                    )
                    for mk in ["INS_BEV", "TD_HF", "BU_HF"]
                }

        # Generate footprint bounding boxes
        inst_bbox_file_path = os.path.join(data_dir, city, "Footprints.pkl")
        # TODO: Uncomment
        # if not os.path.exists(inst_bbox_file_path):
        #     inst_bboxes = get_instance_bboxes(
        #         seg_map.cpu().numpy(),
        #         [INST_RANGES["BLDG"][0], INST_RANGES["CAR"][1]],
        #     )
        #     with open(inst_bbox_file_path, "wb") as fp:
        #         pickle.dump(inst_bboxes, fp)
        # else:
        #     logging.warning("File[Name=%s] exists. Skipping." % inst_bbox_file_path)

        # Generate raycasting results
        raycasting_dir = os.path.join(data_dir, city, "Raycasting")
        os.makedirs(raycasting_dir, exist_ok=True)
        with open(os.path.join(data_dir, city, "CameraRig.json")) as fp:
            cam_rig = json.load(fp)
            cam_rig = cam_rig["cameras"]["CameraComponent"]
            cam_rig["sensor_size"] = [
                get_cfg_value("IMAGE_WIDTH"),
                get_cfg_value("IMAGE_HEIGHT"),
            ]
            # Principal point
            cam_rig["intrinsics"][2] = cam_rig["sensor_size"][0] / 2
            cam_rig["intrinsics"][5] = cam_rig["sensor_size"][1] / 2
            # Focal length
            cam_rig["intrinsics"][0] /= 1920 / get_cfg_value("IMAGE_WIDTH")
            cam_rig["intrinsics"][4] /= 1080 / get_cfg_value("IMAGE_HEIGHT")

        rows = []
        with open(os.path.join(data_dir, city, "CameraPoses.csv")) as fp:
            reader = csv.DictReader(fp)
            rows = [r for r in reader]

        for r in tqdm(rows):
            cam_pose = get_camera_poses(
                r,
                get_cfg_value("BEV_MAP_SIZE") // 2,
                get_cfg_value("SCALE"),
                get_cfg_value("Z_OFFSET"),
            )
            bev_map_bbox = get_bev_map_bbox(cam_pose, get_cfg_value("VOL_SIZE"))
            # Update cam_pose according to the bev_map_bbox
            cam_pose["cam_position"][:2] -= bev_map_bbox["TL"]
            cam_pose["cam_look_at"][:2] -= bev_map_bbox["TL"]
            # Rebuild 3D volume from projection maps
            # TODO: Try to use different scales for different classes
            volume = get_volume_with_scale(
                projections,
                bev_map_bbox,
                {
                    "INST_RANGE": INST_RANGES["BLDG"],
                    "ROOF_HEIGHT": get_cfg_value("BLDG_ROOF_HEIGHT"),
                    "ROOF_OFFSET": get_cfg_value("BLDG_ROOF_OFFSET"),
                    "MAX_HEIGHT": get_cfg_value("MAX_HEIGHT"),
                },
                get_cfg_value("VOL_SIZE"),
            )
            raycasting = get_ray_voxel_intersection(
                cam_rig,
                cam_pose,
                volume,
                get_cfg_value("CLASSES"),
            )
            if is_debug:
                seg_map = utils.helpers.get_seg_map(
                    raycasting["voxel_id"].squeeze()[..., 0].cpu().numpy()
                )
                utils.helpers.get_diffuse_shading_img(
                    seg_map,
                    raycasting["depth2"],
                    raycasting["raydirs"],
                    raycasting["cam_origin"],
                ).save(os.path.join(raycasting_dir, "%04d.png" % int(r["id"])))
            else:
                est_seg_map = Image.open(
                    os.path.join(
                        data_dir, city, seg_map_file_pattern % (city, int(r["id"]))
                    )
                )
                # Change the order of channels for efficiency
                raycasting["depth2"] = raycasting["depth2"].permute(1, 2, 0, 3, 4)
                raycasting = {k: v.cpu().numpy() for k, v in raycasting.items()}
                with open(
                    os.path.join(raycasting_dir, "%04d.pkl" % int(r["id"])), "wb"
                ) as ofp:
                    raycasting["mask"] = get_ambiguous_seg_mask(
                        raycasting["voxel_id"], est_seg_map
                    )
                    pickle.dump(raycasting, ofp)

            # Empty CUDA cache
            del volume
            del raycasting
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.config.dictConfig(
        {
            "disable_existing_loggers": True,
            "format": "[%(levelname)s] %(asctime)s %(message)s",
            "level": logging.DEBUG,
            "version": 1,
        }
    )
    parser = argparse.ArgumentParser(description="The CitySample Dataset Generator")
    parser.add_argument(
        "--data_dir", default=os.path.join(PROJECT_HOME, "data", "city-sample")
    )
    parser.add_argument("--seg_map", default="SemanticImage/%sSequence.%04d.png")
    parser.add_argument("--img_size", default=(960, 540))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.seg_map, args.img_size, args.debug)
