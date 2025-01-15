# -*- coding: utf-8 -*-
#
# @File:   dataset_generator.py
# @Author: Haozhe Xie
# @Date:   2023-12-22 15:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-01-16 06:03:06
# @Email:  root@haozhexie.com

import argparse
import cv2
import csv
import json
import logging
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
    from config import cfg

    CFG_KEYS = {
        "MAX_HEIGHT": "MAX_HEIGHT",
        "BLDG_INST_RANGE": "BLDG.INS_RANGE",
        "CAR_INST_RANGE": "CAR.INS_RANGE",
    }
    CFG_VALUES = {
        "SCALE": 4,
        "Z_OFFSET": 14,  # 14 = -3.5m * scale -> 0 for the water plane
        "VOL_SIZE": 3072,
        "BEV_MAP_SIZE": 19600,
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
        "SIDEWALK": 7,
        "BLDG_FACADE": 8,
        "BLDG_ROOF": 9,
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


def get_projections(city_dir, map_size, z_offset, scale, classes, inst_ranges):
    HOU_SCALE = 4
    assert HOU_SCALE == scale
    # The constants defined in HOU_CLASSES only used in this function.
    HOU_CLASSES = {
        "ROAD": 1,
        "FWY_DECK": 2,
        "FWY_PILLAR": 3,
        "FWY_BARRIER": 4,
        "ZONE": 5,
        "SIDEWALK": 6,
    }
    HOU_INV_INDEX = {v: k for k, v in HOU_CLASSES.items()}
    HOU_SCALES = {
        "ROAD": int(2 * HOU_SCALE),
        "FWY_DECK": int(2 * HOU_SCALE),
        "FWY_PILLAR": int(1 * HOU_SCALE),
        "FWY_BARRIER": int(0.5 * HOU_SCALE),
        "CAR": int(0.25 * HOU_SCALE),
        "ZONE": int(2 * HOU_SCALE),
        "SIDEWALK": int(0.5 * HOU_SCALE),
        "BLDG_FACADE": int(2 * HOU_SCALE),
    }

    points_file_path = os.path.join(city_dir, "Points.pkl")
    if not os.path.exists(points_file_path):
        logging.warning("File not found in %s" % (points_file_path))
        return {}

    with open(points_file_path, "rb") as fp:
        points = pickle.load(fp)

    # Make better alignment with the RGB images
    points[:, :2] -= 1
    # Make all the point coordinates positive at z-axis
    points[:, 2] += z_offset  # - 1

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
    logging.info("Fixing projection holes ...")
    projections["REST"] = _get_water_areas(projections["REST"], classes)
    return projections


def _get_projection(points, map_size, hou_inv_idx, classes, scales, inst_ranges):
    # assert points.dtype == np.int16
    ins_map = np.zeros((map_size, map_size), dtype=points.dtype)
    tpd_hf = np.zeros((map_size, map_size), dtype=points.dtype)
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
    null_area = projection["INS_BEV"] == classes["NULL"]
    _, _, water_area, _ = cv2.floodFill(null_area.astype(np.uint8), None, (0, 0), 1)
    water_area = np.where(water_area[1:-1, 1:-1] == 1)
    projection["INS_BEV"][water_area] = classes["WATER"]
    # Set water plane height to 1 (MAGIC NUMBER)
    projection["TD_HF"][water_area] = 1
    projection["BU_HF"][water_area] = 0

    null_area = projection["INS_BEV"] == classes["NULL"]
    null_area = np.where(null_area)
    projection["INS_BEV"][null_area] = classes["ROAD"]
    # Set road plane height to 14 (MAGIC NUMBER)
    projection["TD_HF"][null_area] = 14
    projection["BU_HF"][null_area] = 13

    return projection


def get_instance_bboxes(projections, inst_range):
    bboxes = {}
    for k, v in projections.items():
        _instances = [
            i
            for i in np.unique(v["INS_BEV"])
            if i >= inst_range[0] and i < inst_range[1]
        ]
        for bi in tqdm(
            _instances, desc="Generating Instance BBoxes[%s]" % k, leave=False
        ):
            bboxes[bi] = cv2.boundingRect((v["INS_BEV"] == bi).astype(np.uint8))

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
    return cam_position + mat3[:3, 0]


def get_bev_map_bbox(projection, cam_rig, cam_pose, inst_bboxes, patch_size, bldg_cfg):
    # The BEV map bounding box is determined by camera positions by default
    patch_center = cam_pose["cam_look_at"][:2]

    # The BEV map bounding box is determined by the major instance in the patch
    if inst_bboxes is not None:
        # Scale the projection maps to the patch size
        scaled_projection = _get_projection_patch(projection, patch_size)
        scale_factor = patch_size / projection["INS_BEV"].shape[0]
        # Adjust the camera position and look-at position
        _cam_pose = {
            "cam_position": cam_pose["cam_position"] * scale_factor,
            "cam_look_at": cam_pose["cam_look_at"] * scale_factor,
        }

        volume = torch.zeros(
            (patch_size, patch_size, int(bldg_cfg["MAX_HEIGHT"] * scale_factor) + 1),
            dtype=torch.int16,
            device=torch.device("cuda:0"),
        )
        volume = extensions.footprint_extruder.extrude_footprint(
            volume,
            scaled_projection["INS_BEV"],
            scaled_projection["TD_HF"],
            scaled_projection["BU_HF"],
            0,
            bldg_cfg["ROOF_HEIGHT"],
            0,
            bldg_cfg["ROOF_OFFSET"],
            bldg_cfg["INST_RANGE"][0],
            bldg_cfg["INST_RANGE"][1],
        )
        raycasting = get_ray_voxel_intersection(cam_rig, _cam_pose, volume)
        voxels = raycasting["voxel_id"][:, :, 0, 0]
        bldg_voxels = voxels[voxels >= bldg_cfg["INST_RANGE"][0]]
        if bldg_voxels.size(0) != 0:
            n_ins_pixels = torch.bincount(bldg_voxels)
            major_inst = torch.argmax(n_ins_pixels).item()
            # Convert Bldg.Roof -> Bldg.Facade
            if major_inst not in inst_bboxes:
                major_inst -= 1
            x, y, w, h = inst_bboxes[major_inst]
            patch_center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        else:
            logging.warning("No building voxels found in the raycasting results.")
            # Fallback to the default patch center
            # patch_center = cam_pose["cam_look_at"][:2]

    # Ordered by: (x, y)
    patch_center = (patch_center + 0.5).astype(np.int32)
    top_left = patch_center - patch_size // 2
    btm_right = patch_center + patch_size // 2
    return {"TL": top_left, "BR": btm_right}


def get_volume_with_scale(projections, bev_map_bbox, bldg_cfg, vol_size):
    volume = torch.zeros(
        (vol_size, vol_size, bldg_cfg["MAX_HEIGHT"]),
        dtype=torch.int16,
        device=torch.device("cuda:0"),
    )
    for k in ["CAR", "FREEWAY", "REST"]:
        _projections = _get_projection_patch(
            projections[k], vol_size, bev_map_bbox, volume.device
        )
        assert torch.min(_projections["TD_HF"]) >= 0
        assert torch.max(_projections["TD_HF"]) < bldg_cfg["MAX_HEIGHT"]

        volume = extensions.footprint_extruder.extrude_footprint(
            volume,
            _projections["INS_BEV"],
            _projections["TD_HF"],
            _projections["BU_HF"],
            0,
            bldg_cfg["ROOF_HEIGHT"],
            0,
            bldg_cfg["ROOF_OFFSET"],
            bldg_cfg["INST_RANGE"][0],
            bldg_cfg["INST_RANGE"][1],
        )
    return volume.squeeze(dim=0)


def _get_projection_patch(projections, patch_size, bev_map_bbox=None, device="cuda:0"):
    INTERPOLATION = {
        "INS_BEV": cv2.INTER_NEAREST,
        "TD_HF": cv2.INTER_LINEAR,
        "BU_HF": cv2.INTER_LINEAR,
    }
    # Crop to patches
    patches = {}
    for k, v in INTERPOLATION.items():
        if bev_map_bbox is not None:
            tl, br = bev_map_bbox["TL"], bev_map_bbox["BR"]
        else:
            tl = [0, 0]
            br = [projections[k].shape[1], projections[k].shape[0]]

        _patch = projections[k][tl[1] : br[1], tl[0] : br[0]].astype(np.int16)
        _scale = 1
        if _patch.shape != (patch_size, patch_size):
            _scale = (
                (patch_size / _patch.shape[0]) + (patch_size / _patch.shape[1])
            ) / 2
            _patch = cv2.resize(_patch, (patch_size, patch_size), interpolation=v)
        # Auto scale the height maps
        if k == "TD_HF" or k == "BU_HF":
            _patch = (_patch * _scale).astype(np.int16)

        patches[k] = utils.helpers.var_or_cuda(
            torch.from_numpy(_patch),
            device,
        )

    return patches


def get_ray_voxel_intersection(cam_rig, cam_pose, volume):
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

    return {
        "voxel_id": voxel_id,
        "depth2": depth2,
        "raydirs": raydirs,
        "viewdir": viewdir,
        "cam_origin": cam_origin,
    }


def get_unambiguous_seg_mask(
    ins_seg_map, est_seg_map, bldg_inst_range, car_inst_range, classes
):
    # Map NULL to WATER
    if "SKY" in classes:
        ins_seg_map[ins_seg_map == 0] = classes["SKY"]

    # NOTE: In ins_seg_map, 4n and 4n+1 denote building facade and roof, respectively.
    #       In est_seg_map, 7 and 8 denote building facade and roof, respectively.
    ins_seg_map[ins_seg_map >= car_inst_range[0]] = classes["CAR"]
    ins_seg_map[(ins_seg_map >= bldg_inst_range[0]) & (ins_seg_map % 4 == 0)] = classes[
        "BLDG_FACADE"
    ]
    ins_seg_map[(ins_seg_map >= bldg_inst_range[0]) & (ins_seg_map % 4 == 1)] = classes[
        "BLDG_ROOF"
    ]
    return ins_seg_map == est_seg_map


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
            logging.info("Generating Projections for %s ..." % city)
            projections = get_projections(
                city_dir,
                get_cfg_value("BEV_MAP_SIZE"),
                get_cfg_value("Z_OFFSET"),
                get_cfg_value("SCALE"),
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
            logging.info("Reading projections for %s ..." % city)
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
        logging.info("Generating footprint bounding boxes for %s ..." % city)
        if not os.path.exists(inst_bbox_file_path):
            inst_bboxes = get_instance_bboxes(
                projections,
                [INST_RANGES["BLDG"][0], INST_RANGES["CAR"][1]],
            )
            with open(inst_bbox_file_path, "wb") as fp:
                pickle.dump(inst_bboxes, fp)
        else:
            logging.warning("File[Name=%s] exists. Skipping." % inst_bbox_file_path)
            with open(inst_bbox_file_path, "rb") as fp:
                inst_bboxes = pickle.load(fp)

        # Generate raycasting results
        raycasting_dir = os.path.join(data_dir, city, "Raycasting")
        os.makedirs(raycasting_dir, exist_ok=True)
        with open(os.path.join(data_dir, city, "CameraRig.json")) as fp:
            cam_rig = json.load(fp)
            cam_rig = cam_rig["cameras"]["CameraComponent"]
            cam_rig["sensor_size"] = img_size
            # Principal point
            cam_rig["intrinsics"][2] = cam_rig["sensor_size"][0] / 2
            cam_rig["intrinsics"][5] = cam_rig["sensor_size"][1] / 2
            # Focal length
            cam_rig["intrinsics"][0] /= 1920 / img_size[0]
            cam_rig["intrinsics"][4] /= 1080 / img_size[1]

        rows = []
        with open(os.path.join(data_dir, city, "CameraPoses.csv")) as fp:
            reader = csv.DictReader(fp)
            rows = [r for r in reader]

        bldg_cfg = {
            "INST_RANGE": INST_RANGES["BLDG"],
            "ROOF_HEIGHT": get_cfg_value("BLDG_ROOF_HEIGHT"),
            "ROOF_OFFSET": get_cfg_value("BLDG_ROOF_OFFSET"),
            "MAX_HEIGHT": get_cfg_value("MAX_HEIGHT"),
        }
        for r in tqdm(rows):
            cam_pose = get_camera_poses(
                r,
                get_cfg_value("BEV_MAP_SIZE") // 2,
                get_cfg_value("SCALE"),
                get_cfg_value("Z_OFFSET"),
            )
            bev_map_bbox = get_bev_map_bbox(
                projections["REST"],
                cam_rig,
                cam_pose,
                inst_bboxes if city != "City00" else None,
                get_cfg_value("VOL_SIZE"),
                bldg_cfg,
            )
            # Update cam_pose according to the bev_map_bbox
            cam_pose["cam_position"][:2] -= bev_map_bbox["TL"]
            cam_pose["cam_look_at"][:2] -= bev_map_bbox["TL"]
            # Rebuild 3D volume from projection maps
            volume = get_volume_with_scale(
                projections, bev_map_bbox, bldg_cfg, get_cfg_value("VOL_SIZE")
            )
            raycasting = get_ray_voxel_intersection(cam_rig, cam_pose, volume)
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
                est_seg_map = cv2.resize(
                    np.array(est_seg_map.convert("P")),
                    (img_size[0], img_size[1]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Change the order of channels for efficiency
                raycasting["depth2"] = raycasting["depth2"].permute(1, 2, 0, 3, 4)
                raycasting = {k: v.cpu().numpy() for k, v in raycasting.items()}
                with open(
                    os.path.join(raycasting_dir, "%04d.pkl" % int(r["id"])), "wb"
                ) as ofp:
                    bev_map_center = (bev_map_bbox["BR"] + bev_map_bbox["TL"]) / 2 + 0.5
                    raycasting["img_center"] = {
                        "cx": int(bev_map_center[0]),
                        "cy": int(bev_map_center[1]),
                    }
                    raycasting["mask"] = get_unambiguous_seg_mask(
                        raycasting["voxel_id"][:, :, 0, 0].copy(),
                        est_seg_map,
                        get_cfg_value("BLDG_INST_RANGE"),
                        get_cfg_value("CAR_INST_RANGE"),
                        get_cfg_value("CLASSES"),
                    )
                    pickle.dump(raycasting, ofp)

            # Empty CUDA cache
            del volume
            del raycasting
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
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
