# -*- coding: utf-8 -*-
#
# @File:   traffic_scenario_generator.py
# @Author: Haozhe Xie
# @Date:   2024-11-02 15:17:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-11-06 13:59:48
# @Email:  root@haozhexie.com

import argparse
import cv2
import itertools
import logging
import numpy as np
import os
import pickle
import queue
import skimage.morphology
import sys
import torch

from PIL import Image
from tqdm import tqdm

# Disable the warning message for PIL decompression bomb
# Ref: https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import extensions.keypoint_detector
import utils.helpers

# X, Y offsets for 8-connected grid markers.
# The neighbors are marked as follows:
#  1  2   4
#  8  X  16
# 32 64 128
KPT_GRID_MARKERS = {
    1: (-1, -1),
    2: (0, -1),
    4: (1, -1),
    8: (-1, 0),
    16: (1, 0),
    32: (-1, 1),
    64: (0, 1),
    128: (1, 1),
}


def get_cfg_values(key):
    from config import cfg

    return cfg.DATASETS.CITY_SAMPLE[key]


def get_road_networks(projection_dir, project_names):
    classes = get_cfg_values("CLASSES")
    projections = {
        pn: np.array(Image.open(os.path.join(projection_dir, "%s_INS_BEV.png" % pn)))
        for pn in project_names
    }
    for k, v in projections.items():
        v[v != classes["ROAD"]] = classes["NULL"]
        # Fix holes in the road network
        v = cv2.dilate(v.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
        projections[k] = cv2.erode(v, np.ones((5, 5), np.uint8), iterations=2)

    # Debug: Visualization
    # import utils.helpers
    # utils.helpers.get_seg_map(projections["REST"]).save("output/test.png")
    # import pdb; pdb.set_trace()
    return projections


def _get_road_edges(road_net):
    return cv2.Canny(road_net * 255, 50, 150).astype(bool)


def _get_road_centers(road_net):
    # return skimage.morphology.medial_axis(road_net, return_distance=True)
    return skimage.morphology.skeletonize(road_net).astype(bool)


def get_traffic_maps(road_networks):
    traffic_maps = {}
    for k, v in road_networks.items():
        traffic_maps[k] = {
            "EDGE": _get_road_edges(v),
            "CNTR": _get_road_centers(v),
        }

    return traffic_maps


@utils.helpers.static_vars(
    kpts_values=[a + b for a, b in itertools.combinations(KPT_GRID_MARKERS.keys(), 2)]
)
def _get_starting_node(kpts_map, kp_xs, kp_ys, is_closed):
    for x, y in zip(kp_xs, kp_ys):
        if is_closed:
            return x, y
        elif kpts_map[y, x] not in _get_starting_node.kpts_values:
            # Ensure that the point connects to a number of edges that is not equal to 2.
            return x, y

    assert False, "No starting node found!"


def _get_next_ngr_nodes(kpts_map, curr_node):
    kpt_value = kpts_map[curr_node[1], curr_node[0]]
    ngr_node_offsets = [v for k, v in KPT_GRID_MARKERS.items() if k & kpt_value]

    next_ngr_nodes = []
    for nno in ngr_node_offsets:
        _curr_node = curr_node
        # Make sure the next node is within the map
        while True:
            _curr_node = _curr_node[0] + nno[0], _curr_node[1] + nno[1]
            if (
                _curr_node[0] < 0
                or _curr_node[0] >= kpts_map.shape[1]
                or _curr_node[1] < 0
                or _curr_node[1] >= kpts_map.shape[0]
            ):
                break
            if kpts_map[_curr_node[1], _curr_node[0]] != 0:
                next_ngr_nodes.append(_curr_node)
                break

    return next_ngr_nodes


def _get_nodes(kpts_map, kp_xs, kp_ys, closed):
    nodes = {}  # Key: Node Coordinates, Value: Node ID
    unvisited_nodes = queue.Queue()
    # DFS
    unvisited_nodes.put(_get_starting_node(kpts_map, kp_xs, kp_ys, closed))
    while not unvisited_nodes.empty():
        curr_node = unvisited_nodes.get()
        next_nodes = _get_next_ngr_nodes(kpts_map, curr_node)
        nodes[curr_node] = {
            "value": kpts_map[curr_node[1], curr_node[0]],
            "next": next_nodes,
        }
        for nn in next_nodes:
            if nn not in nodes:
                unvisited_nodes.put(nn)

    # Remove the vertex of right-angle inflection point.
    # This point should not be regarded as a keypoint.
    _nodes = nodes.copy()
    for k, v in _nodes.items():
        if v["value"] in [10, 18, 72, 80]:
            del nodes[k]
    for k, v in nodes.items():
        v["next"] = [nn for nn in v["next"] if nn in nodes]

    return nodes


def _get_ways(nodes, closed):
    ways = []
    edges = set()
    unvisited_edges = queue.Queue()
    # Determine the starting edge
    next_nodes = [k for k, v in nodes.items() if len(v["next"]) != 2]
    node_start = next(iter(next_nodes)) if next_nodes else list(nodes.keys())[0]
    for nn in nodes[node_start]["next"]:
        unvisited_edges.put((node_start, nn))

    while not unvisited_edges.empty():
        node_prev, node_next = unvisited_edges.get()
        node_start = node_prev
        if (node_prev, node_next) in edges or (node_next, node_prev) in edges:
            continue

        edges.add((node_prev, node_next))
        _way = [node_prev, node_next]
        while len(nodes[node_next]["next"]) == 2 and node_next != node_start:
            next_nodes = [nnn for nnn in nodes[node_next]["next"] if nnn != node_prev]
            if not next_nodes:
                break

            node_next_next = next(iter(next_nodes))
            edges.add((node_next, node_next_next))
            _way.append(node_next_next)
            node_prev, node_next = node_next, node_next_next

        # Force the way to be closed
        if node_next != node_start and closed:
            _way.append(node_start)

        ways.append(_way)
        if len(nodes[node_next]["next"]) > 2:
            for nn in nodes[node_next]["next"]:
                unvisited_edges.put((node_next, nn))

    return ways


def _get_kpts_graph(skeleton, closed=True):
    kpts_map = (
        extensions.keypoint_detector.detect_keypoints(torch.from_numpy(skeleton).cuda())
        .cpu()
        .numpy()
    )
    n_conn, conn_map = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )
    for i in tqdm(range(1, n_conn)):
        # TODO: Threshold for small connected components
        _kpts_map = kpts_map * (conn_map == i)
        kp_ys, kp_xs = np.where(_kpts_map != 0)  # ([ys], [xs])
        _nodes = _get_nodes(_kpts_map, kp_xs, kp_ys, closed)
        _ways = _get_ways(_nodes, closed)


def get_traffic_graphs(traffic_maps):
    traffic_graphs = {}
    for tk, tv in traffic_maps.items():
        print(tk)
        traffic_graphs[tk] = {
            # "EDGE": _get_kpts_graph(tv["EDGE"]),
            "CNTR": _get_kpts_graph(tv["CNTR"], closed=False),
        }
    return traffic_graphs


def main(projection_dir, project_names):
    # logging.info("Parsing Road Networks ...")
    # road_networks = get_road_networks(projection_dir, project_names)
    # logging.info("Parsing Traffic Maps ...")
    # traffic_maps = get_traffic_maps(road_networks)
    with open("output/traffic_maps.pkl", "rb") as f:
        traffic_maps = pickle.load(f)

    logging.info("Parsing Traffic Graphs ...")
    traffic_graphs = get_traffic_graphs(traffic_maps)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="The Traffic Scenario Generator")
    parser.add_argument(
        "--projection_dir",
        default=os.path.join(
            PROJECT_HOME, "data", "city-sample", "City01", "Projections"
        ),
    )
    parser.add_argument("--project_names", default="REST, FREEWAY")
    args = parser.parse_args()
    main(
        args.projection_dir,
        [pn.strip() for pn in args.project_names.split(",")],
    )
