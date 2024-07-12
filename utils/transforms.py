# -*- coding: utf-8 -*-
#
# @File:   transforms.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 14:18:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-07-12 21:10:50
# @Email:  root@haozhexie.com

import cv2
import numpy as np
import torch

import utils.helpers


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            if tr is None:
                continue

            transformer = eval(tr["callback"])
            parameters = tr["parameters"] if "parameters" in tr else None
            self.transformers.append(
                {
                    "callback": transformer(
                        parameters, tr["objects"] if "objects" in tr else None
                    ),
                }
            )

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr["callback"]
            data = transform(data)

        return data


class ToTensor(object):
    def __init__(self, _, objects):
        self.objects = objects

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                if len(v.shape) == 2:
                    # H, W -> H, W, C
                    v = v[..., None]
                if len(v.shape) == 3:
                    # H, W, C -> C, H, W
                    v = v.transpose((2, 0, 1))

                data[k] = torch.from_numpy(v).float()

        return data


class RandomInstances(object):
    """Randomly select an instance (buildings or cars) from the visible instances."""

    def __init__(self, parameters, objects):
        self.instances = parameters["instances"] if "instances" in parameters else None
        # NOTE: For BLDG, the roof instance is the next to the facade instance, i.e., cont_instances = 1.
        self.cont_instances = (
            parameters["cont_instances"] if "cont_instances" in parameters else []
        )
        self.objects = objects

    def __call__(self, data):
        ins_map = data["voxel_id"][..., 0, 0] * data["mask"]
        visible_ins = np.unique(ins_map[np.isin(ins_map, self.instances)])

        if len(visible_ins) == 0:
            return data

        data["inst"] = [np.random.choice(visible_ins)]
        for ci in self.cont_instances:
            data["inst"].append(data["inst"][0] + ci)

        ins_mask = np.isin(ins_map, data["inst"])
        data["mask"] &= ins_mask
        return data


class RandomCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.mode = parameters["mode"] if "mode" in parameters else "random"
        self.n_min_pixels = (
            parameters["n_min_pixels"] if "n_min_pixels" in parameters else 0
        )
        self.objects = objects

    def _get_offsets(self, image_w, image_h, patch_w, patch_h, data):
        if self.mode in ["random", "center"]:
            offset_x = self._get_offset(image_w, patch_w)
            offset_y = self._get_offset(image_h, patch_h)
        elif self.mode == "instance":
            x, y = self._get_instance_bbox(np.isin(data["voxel_id"][..., 0, 0], data["inst"]))
            cx, cy = np.random.randint(x[0], x[1]), np.random.randint(y[0], y[1])
            offset_x = min(max(0, cx - patch_w // 2), image_w - patch_w)
            offset_y = min(max(0, cy - patch_h // 2), image_h - patch_h)
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

        return offset_x, offset_y

    def _get_offset(self, size, crop_size):
        if size == crop_size:
            return 0
        elif self.mode == "random":
            return np.random.randint(0, size - crop_size - 1)
        elif self.mode == "center":
            return size // 2 - crop_size // 2

    def _get_instance_bbox(self, ins_mask):
        # https://github.com/hzxie/CityDreamer/blob/master/utils/transforms.py?ref_type=heads#L138
        pts = cv2.findNonZero(ins_mask.astype(np.uint8))
        x_min, x_max = np.min(pts[..., 0]), np.max(pts[..., 0])
        y_min, y_max = np.min(pts[..., 1]), np.max(pts[..., 1])
        return (x_min, x_max), (y_min, y_max)

    def _get_img_patch(self, img, offset_x, offset_y):
        return img[offset_y : offset_y + self.height, offset_x : offset_x + self.width]

    def _get_crop_position(self, data, width, height):
        N_MAX_TRY_TIMES = 100
        img = data[self.objects[0]]
        ih, iw = img.shape[0], img.shape[1]
        # Check the cropped patch contains enough informative pixels for training
        for _ in range(N_MAX_TRY_TIMES):
            offset_x, offset_y = self._get_offsets(iw, ih, width, height, data)
            mask = self._get_img_patch(data["mask"], offset_x, offset_y)

            n_pixels = np.count_nonzero(mask)
            if n_pixels >= self.n_min_pixels:
                break

        return offset_x, offset_y, mask

    def __call__(self, data):
        width, height = self.width, self.height
        offset_x, offset_y = None, None
        while offset_x is None or offset_y is None:
            offset_x, offset_y, mask = self._get_crop_position(data, width, height)

        # Crop all data fields simultaneously
        data["crp"] = {
            "x": offset_x,
            "y": offset_y,
            "w": self.width,
            "h": self.height,
        }
        for k, v in data.items():
            if k == "mask":
                # Prevent duplicated computation
                data[k] = mask
            if k in self.objects:
                data[k] = self._get_img_patch(v, offset_x, offset_y)

        return data


class BevResize(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _get_resized_img(self, img, width, height):
        return cv2.resize(img, (width, height))

    def __call__(self, data):
        for k in self.objects:
            data[k] = self._get_resized_img(data[k], self.width, self.height)

        return data


class BevCrop(object):
    def __init__(self, parameters, objects):
        self.height = parameters["height"]
        self.width = parameters["width"]
        self.objects = objects

    def _get_img_patch(self, img, cx, cy, half_width, half_height):
        tl_x, br_x = cx - half_width, cx + half_width
        tl_y, br_y = cy - half_height, cy + half_height
        return img[tl_y:br_y, tl_x:br_x]

    def __call__(self, data):
        # In instance mode, the center is determined by the cx, cy of the instance.
        # Otherwise, the center is determined by the camera position / look at position.
        instance_mode = "inst" in data
        cx, cy = data["img_center"]["cx"], data["img_center"]["cy"]
        if instance_mode:
            assert type(data["inst"]) == list
            inst = data["inst"][0]
            # https://github.com/hzxie/city-dreamer/blob/master/utils/datasets.py?ref_type=heads#L489
            dx, dy, w, h = data["building_stats"][inst]
            data["building_stat"] = torch.Tensor([dy, dx, h, w, inst])
            cx = int(cx + data["building_stat"][1])
            cy = int(cy + data["building_stat"][0])

        for k in self.objects:
            data[k] = self._get_img_patch(
                data[k], cx, cy, self.width // 2, self.height // 2
            )

        return data


class InstanceToSemantic(object):
    def __init__(self, parameters, objects):
        self.semantic_classes = parameters["semantic_classes"]
        self.min_instances = parameters["min_instances"]
        self.objects = objects

    def _instances_to_semantic(self, ins_map, mapper):
        if mapper is not None:
            # Instance Mode: the specific building instance are mapped to its semantic label
            for src, dst in mapper.items():
                ins_map[ins_map == src] = dst
            # The rest instances are set to NULL
            ins_map[ins_map >= self.min_instances] = 0
        else:
            # Background Mode: all instances are set to their semantic classes.
            for sc in self.semantic_classes.values():
                selector = (ins_map >= sc["cond"]["range"][0]) & (
                    ins_map < sc["cond"]["range"][1]
                )
                ins_map[selector] = sc["smtc"]

        return ins_map

    def __call__(self, data):
        # In instance mode, only the selected instance is kept. The rest are set to NULL.
        # Otherwise, all instances are set to their semantic classes.
        instance_mode = "inst" in data
        mapper = None
        if instance_mode:
            assert type(data["inst"]) == list
            mapper = {}
            for i in data["inst"]:
                for sc in self.semantic_classes.values():
                    if (
                        i >= sc["cond"]["range"][0]
                        and i < sc["cond"]["range"][1]
                        and sc["cond"]["cond"](i)
                    ):
                        mapper[i] = sc["smtc"]

        for k, v in data.items():
            if k in self.objects:
                data[k] = self._instances_to_semantic(v, mapper)

        return data


class MaskRaydirs(object):
    def __init__(self, parameters, objects):
        self.parameters = parameters
        self.objects = objects

    def __call__(self, data):
        assert "inst" in data, "RandomInstance should be executed before MaskRaydirs."
        seg_map = data["voxel_id"][..., 0, 0]
        mask = np.isin(seg_map, data["inst"])
        data["raydirs"][~mask] = 0
        return data


class ToOneHot(object):
    def __init__(self, parameters, objects):
        self.n_classes = parameters["n_classes"]
        self.ignored_classes = (
            parameters["ignored_classes"] if "ignored_classes" in parameters else []
        )
        self.objects = objects

    def _to_onehot(self, img):
        mask = utils.helpers.mask_to_onehot(img, self.n_classes, self.ignored_classes)
        return mask

    def __call__(self, data):
        for k, v in data.items():
            if k in self.objects:
                data[k] = self._to_onehot(v)

        return data
