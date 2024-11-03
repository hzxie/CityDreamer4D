# -*- coding: utf-8 -*-
#
# @File:   ue_keyframe_generator.py
# @Author: Haozhe Xie
# @Date:   2024-08-29 21:25:09
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-11-03 18:34:34
# @Email:  root@haozhexie.com

import argparse
import csv
import logging
import numpy as np
import os
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
import scripts.dataset_generator as dg


def get_cfg_value(key):
    value = dg.get_cfg_value(key)
    if value is not None:
        return value

    CFG_VALUES = {
        "N_KEY_FRAMES": 3000,
        "N_VIEWPOINTS": 36,
        "MIN_VISIBLE_INSTANCES": 10,
        "MIN_BLDG_PIXELS": int(960 * 540 * 0.5),
        "PITCH_RANGE": [-75, 30],
        "IMG_SIZE": (960, 540),
        "FOCAL_LENGTH": 1414.1415820071118,
    }
    CFG_VALUES["CAM_RIG"] = {
        "intrinsics": [
            CFG_VALUES["FOCAL_LENGTH"],
            0,
            CFG_VALUES["IMG_SIZE"][0],
            0,
            CFG_VALUES["FOCAL_LENGTH"],
            CFG_VALUES["IMG_SIZE"][1],
            0,
            0,
            1,
        ],
        "sensor_size": CFG_VALUES["IMG_SIZE"],
    }
    return CFG_VALUES[key] if key in CFG_VALUES else None


def get_scaled_projections(projections, patch_size, classes):
    scaled_projections = {}
    # Scale the projection maps to the patch size
    for k, v in projections.items():
        scaled_projections[k] = dg._get_projection_patch(v, patch_size)

    td_hf = scaled_projections["REST"]["TD_HF"]
    ins_bev = scaled_projections["REST"]["INS_BEV"]
    zone_area = torch.isin(
        ins_bev, torch.tensor([classes["ROAD"], classes["ZONE"]], device=ins_bev.device)
    )
    # Fix misalignment in the height field during BEV map resize
    td_hf[zone_area] = 2
    return scaled_projections


def get_bev_map_bbox(projection, classes):
    bev_map = projection["INS_BEV"]
    x, y = torch.where(
        ~torch.isin(
            bev_map,
            torch.tensor([classes["NULL"], classes["WATER"]], device=bev_map.device),
        )
    )
    z_min = torch.min(projection["TD_HF"][projection["TD_HF"] > 1]).item() + 1
    z_max = torch.max(projection["TD_HF"]).item()
    return (
        (x.min().item(), x.max().item()),
        (y.min().item(), y.max().item()),
        (z_min, z_max),
    )


def get_volume(projections, bldg_cfg):
    h, w = projections["REST"]["INS_BEV"].shape
    d = torch.max(projections["REST"]["TD_HF"]).item() + 1
    volume = torch.zeros(
        (h, w, d),
        dtype=torch.int16,
        device=torch.device("cuda:0"),
    )
    for p in projections.values():
        volume = extensions.footprint_extruder.extrude_footprint(
            volume,
            p["INS_BEV"],
            p["TD_HF"],
            p["BU_HF"],
            0,
            bldg_cfg["ROOF_HEIGHT"],
            0,
            bldg_cfg["ROOF_OFFSET"],
            bldg_cfg["INST_RANGE"][0],
            bldg_cfg["INST_RANGE"][1],
        )
    return volume


def get_keyframes(
    bev_map_bbox,
    cam_rig,
    volume,
    n_viewpoints,
    min_visible_instances,
    min_bldg_pixels,
    pitch_range,
    cam_altitude_range,
    bldg_cfg,
):
    cam_position = [
        np.random.uniform(bev_map_bbox[0][0], bev_map_bbox[0][1]),
        np.random.uniform(bev_map_bbox[1][0], bev_map_bbox[1][1]),
        np.random.uniform(cam_altitude_range[0], cam_altitude_range[1]),
    ]
    pitch = np.random.uniform(pitch_range[0], pitch_range[1])

    keyframes = []
    for i in range(n_viewpoints):
        yaw = 360.0 / n_viewpoints * i
        cam_pose = {
            "cam_position": cam_position,
            "cam_look_at": _get_cam_look_at(cam_position, yaw, pitch),
        }
        raycasting = dg.get_ray_voxel_intersection(cam_rig, cam_pose, volume)
        instances = torch.unique(raycasting["voxel_id"])

        seg_map = raycasting["voxel_id"].squeeze()[..., 0]
        seg_map[
            (seg_map >= bldg_cfg["INST_RANGE"][0])
            & (seg_map < bldg_cfg["INST_RANGE"][1])
        ] = bldg_cfg["FACADE_CID"]
        n_bldg_pixels = torch.count_nonzero(seg_map == bldg_cfg["FACADE_CID"])
        # print(i, len(instances), cam_pose, yaw)

        if len(instances) >= min_visible_instances and n_bldg_pixels >= min_bldg_pixels:
            keyframes.append(
                {
                    "tx": cam_position[0],
                    "ty": cam_position[1],
                    "tz": cam_position[2],
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": 0,
                }
            )

        # # Debug: Visualize the raycasting results
        # import utils.helpers
        # utils.helpers.get_diffuse_shading_img(
        #     seg_map,
        #     raycasting["depth2"],
        #     raycasting["raydirs"],
        #     raycasting["cam_origin"],
        # ).save(os.path.join("output/frames/%04d.png" % i))

    return keyframes


def _get_cam_look_at(cam_position, yaw, pitch):
    tan_pitch = np.tan(np.radians(pitch))
    radius = cam_position[2] / abs(tan_pitch)
    x = cam_position[0] + radius * np.cos(np.radians(yaw))
    y = cam_position[1] + radius * np.sin(np.radians(yaw))
    z = cam_position[2] * 2 if tan_pitch > 0 else 0
    return [x, y, z]


def main(data_dir):
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
            projections = dg.get_projections(
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

        proj_scale = projections["REST"]["INS_BEV"].shape[0] / get_cfg_value("VOL_SIZE")
        proj_scale = proj_scale / get_cfg_value("SCALE") * 100
        projections = get_scaled_projections(
            projections, get_cfg_value("VOL_SIZE"), get_cfg_value("CLASSES")
        )
        bev_map_bbox = get_bev_map_bbox(
            projections["CAR"] if city == "City00" else projections["REST"],
            get_cfg_value("CLASSES"),
        )

        # Generate seg volume
        seg_volume = get_volume(
            projections,
            {
                "INST_RANGE": INST_RANGES["BLDG"],
                "ROOF_HEIGHT": get_cfg_value("BLDG_ROOF_HEIGHT"),
                "ROOF_OFFSET": get_cfg_value("BLDG_ROOF_OFFSET"),
            },
        )

        # Generate keyframes
        cam_rig = get_cfg_value("CAM_RIG")
        keyframes = []
        logging.info("Generating KeyFrames for %s ..." % city)
        pbar = tqdm(total=get_cfg_value("N_KEY_FRAMES"))
        while len(keyframes) < get_cfg_value("N_KEY_FRAMES"):
            _keyframes = get_keyframes(
                bev_map_bbox,
                cam_rig,
                seg_volume,
                get_cfg_value("N_VIEWPOINTS"),
                get_cfg_value("MIN_VISIBLE_INSTANCES"),
                get_cfg_value("MIN_BLDG_PIXELS") if city != "City00" else 0,
                get_cfg_value("PITCH_RANGE"),
                bev_map_bbox[2],
                {
                    "INST_RANGE": INST_RANGES["BLDG"],
                    "FACADE_CID": get_cfg_value("CLASSES")["BLDG_FACADE"],
                },
            )
            # Convert the camera position and make it matches the UE 5 coordinate system
            for kf in _keyframes:
                kf["tx"] = (kf["tx"] - get_cfg_value("VOL_SIZE") // 2) * proj_scale
                kf["ty"] = (kf["ty"] - get_cfg_value("VOL_SIZE") // 2) * proj_scale
                kf["tz"] = kf["tz"] * proj_scale
                keyframes.append(kf)

            pbar.update(len(_keyframes))

        # Save keyframes
        with open(os.path.join(city_dir, "KeyFrames.csv"), "w", newline="") as csvfile:
            fieldnames = keyframes[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(keyframes)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="The CitySample Dataset KeyFrame Generator"
    )
    parser.add_argument(
        "--data_dir", default=os.path.join(PROJECT_HOME, "data", "city-sample")
    )
    args = parser.parse_args()
    main(args.data_dir)
