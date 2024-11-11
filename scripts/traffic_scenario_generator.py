# -*- coding: utf-8 -*-
#
# @File:   traffic_scenario_generator.py
# @Author: Haozhe Xie
# @Date:   2024-11-02 15:17:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-11-11 21:15:15
# @Email:  root@haozhexie.com

import argparse
import cv2
import logging
import numpy as np
import os
import pickle
import queue
import scipy.ndimage
import shapely
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
        # v = cv2.dilate(v.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
        # projections[k] = cv2.erode(v, np.ones((5, 5), np.uint8), iterations=2)
        v = scipy.ndimage.gaussian_filter(v * 255, sigma=5)
        projections[k] = (v >= 128).astype(np.uint8)

    # Debug: Visualization
    # import utils.helpers
    # utils.helpers.get_seg_map(projections["REST"]).save("output/test.png")
    return projections


def _get_road_edges(road_net):
    return cv2.Canny(road_net * 255, 50, 150).astype(bool)


def _get_road_centers(road_net):
    # return skimage.morphology.medial_axis(road_net, return_distance=True)
    # return skimage.morphology.skeletonize(road_net).astype(bool)
    skeleton = cv2.ximgproc.thinning(road_net * 255)
    return skeleton.astype(bool)


def get_traffic_maps(road_networks):
    traffic_maps = {}
    for k, v in road_networks.items():
        traffic_maps[k] = {
            "EDGE": _get_road_edges(v),
            "CNTR": _get_road_centers(v),
        }
    return traffic_maps


def _get_ngr_node_offsets(kpt_value):
    # X, Y offsets for 8-connected grid markers.
    # The neighbors are marked as follows:
    #  1  2   4
    #  8  X  16
    # 32 64 128
    # fmt: off
    KPT_GRID_MARKERS = {
        0x01: (-1, -1), # 1
        0x02: (0, -1),  # 2
        0x04: (1, -1),  # 4
        0x08: (-1, 0),  # 8
        0x10: (1, 0),   # 16
        0x20: (-1, 1),  # 32
        0x40: (0, 1),   # 64
        0x80: (1, 1),   # 128
    }
    # fmt: on
    offsets = []
    # Non-diagonal
    for kgm in [0x02, 0x08, 0x10, 0x40]:
        if kpt_value & kgm:
            offsets.append(KPT_GRID_MARKERS[kgm])
    # Diagonal
    if kpt_value & 0x01 and not (kpt_value & 0x02 or kpt_value & 0x08):
        offsets.append(KPT_GRID_MARKERS[0x01])
    if kpt_value & 0x04 and not (kpt_value & 0x02 or kpt_value & 0x10):
        offsets.append(KPT_GRID_MARKERS[0x04])
    if kpt_value & 0x20 and not (kpt_value & 0x08 or kpt_value & 0x40):
        offsets.append(KPT_GRID_MARKERS[0x20])
    if kpt_value & 0x80 and not (kpt_value & 0x10 or kpt_value & 0x40):
        offsets.append(KPT_GRID_MARKERS[0x80])

    return offsets


def _get_starting_node(kpts_map, kp_xs, kp_ys, is_closed):
    for x, y in zip(kp_xs, kp_ys):
        if is_closed:
            return x, y
        elif len(_get_ngr_node_offsets(kpts_map[y, x])) != 2:
            # Ensure that the point connects to a number of edges that is not equal to 2.
            return x, y

    assert False, "No starting node found!"


def _get_next_ngr_nodes(kpts_map, curr_node):
    kpt_value = kpts_map[curr_node[1], curr_node[0]]
    ngr_node_offsets = _get_ngr_node_offsets(kpt_value)

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
    # Node Attrs: Key: Node Coordinates, Value: {"value": int, "next": list}
    nodes = {}
    unvisited_nodes = queue.Queue()
    # DFS
    unvisited_nodes.put(_get_starting_node(kpts_map, kp_xs, kp_ys, closed))
    while not unvisited_nodes.empty():
        curr_node = unvisited_nodes.get()
        if curr_node in nodes:
            continue

        next_nodes = _get_next_ngr_nodes(kpts_map, curr_node)
        nodes[curr_node] = {
            "value": kpts_map[curr_node[1], curr_node[0]],
            "next": next_nodes,
        }
        for nn in next_nodes:
            if nn not in nodes:
                unvisited_nodes.put(nn)

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


def _simplify_ways(ways, tolerance=12):
    for i, way in enumerate(tqdm(ways, desc="Simplifying ways", leave=False)):
        if len(way) < 2:
            continue

        is_closed = way[0] == way[-1]
        if is_closed:
            shape = shapely.simplify(shapely.Polygon(way), tolerance=tolerance)
            x, y = shape.exterior.coords.xy
        else:
            shape = shapely.simplify(shapely.LineString(way), tolerance=tolerance)
            x, y = shape.coords.xy

        ways[i] = {
            "nodes": np.array([np.array(x), np.array(y)]).astype(np.int32).T,
        }
    return ways


def _get_kpts_graph(skeleton, closed):
    kpts_map = (
        extensions.keypoint_detector.detect_keypoints(torch.from_numpy(skeleton).cuda())
        .cpu()
        .numpy()
    )
    n_conn, conn_map = cv2.connectedComponents(
        skeleton.astype(np.uint8), connectivity=8
    )

    graphs = []
    for i in tqdm(range(1, n_conn), desc="Pasring connected graphs", leave=False):
        _kpts_map = kpts_map * (conn_map == i)
        kp_ys, kp_xs = np.where(_kpts_map != 0)  # ([ys], [xs])
        _nodes = _get_nodes(_kpts_map, kp_xs, kp_ys, closed)
        _ways = _get_ways(_nodes, closed)
        _ways = _simplify_ways(_ways)
        graphs.append(_ways)

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for ways in graphs:
    #     for way in ways:
    #         img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    return graphs


def _get_way_nodes(graphs):
    nodes = {}
    ways = {}
    # Organzing connectivity
    for graph in graphs:
        for way in graph:
            way_id = len(ways)
            way_nodes = [tuple(n) for n in way["nodes"]]
            end_nodes = [way_nodes[0], way_nodes[-1]]
            ways[way_id] = {"id": way_id, "nodes": way_nodes}
            for en in end_nodes:
                if en not in nodes:
                    nodes[en] = {"ways": []}
                nodes[en]["ways"].append(way_id)

    return nodes, ways


def _get_way_length(way_nodes):
    length = 0
    for i in range(1, len(way_nodes)):
        length += np.linalg.norm(np.array(way_nodes[i]) - np.array(way_nodes[i - 1]))

    return length


def _connect_ways(ways, ways_to_connect):
    assert len(ways_to_connect) <= 3

    for shared_way, wc in ways_to_connect:
        way_nodes = [*ways[shared_way]["nodes"]]
        _ways = list(wc)
        for w in _ways:
            if w == shared_way:
                continue
            if way_nodes[0] == ways[w]["nodes"][0]:
                way_nodes = list(reversed(ways[w]["nodes"][1:])) + way_nodes
            elif way_nodes[0] == ways[w]["nodes"][-1]:
                way_nodes = ways[w]["nodes"] + way_nodes[1:]
            elif way_nodes[-1] == ways[w]["nodes"][0]:
                way_nodes = way_nodes + ways[w]["nodes"][1:]
            elif way_nodes[-1] == ways[w]["nodes"][-1]:
                way_nodes = way_nodes + list(reversed(ways[w]["nodes"][:-1]))
            else:
                raise ValueError(
                    "The shared way is not connected to the way to connect."
                )
            # Remove the original way
            del ways[w]
        # Update the way nodes of the shared way
        ways[shared_way]["nodes"] = way_nodes

    return ways


def _remove_short_loops(graphs):
    nodes, ways = _get_way_nodes(graphs)
    endnodes = {}
    # Check if there are two ways that share the same end nodes.
    for k, v in ways.items():
        _end_nodes = (v["nodes"][0], v["nodes"][-1])
        if _end_nodes not in endnodes:
            endnodes[_end_nodes] = []
        endnodes[_end_nodes].append(k)

    ways_to_remove = []
    ways_to_connect = []
    for _endnodes, _ways in endnodes.items():
        if len(_ways) < 2:
            continue

        way_length = {w: _get_way_length(ways[w]["nodes"]) for w in _ways}
        shortest_way = min(way_length, key=way_length.get)
        ways_to_remove.extend([w for w in _ways if w != shortest_way])
        # Connect the rest of the ways connected to the shortest way
        ways_to_connect.append(
            (
                shortest_way,
                set(
                    [
                        w
                        for w in nodes[_endnodes[0]]["ways"]
                        + nodes[_endnodes[1]]["ways"]
                        if w not in ways_to_remove
                    ]
                ),
            )
        )

    ways = _connect_ways(
        {k: v for k, v in ways.items() if k not in ways_to_remove}, ways_to_connect
    )
    return [list(ways.values())]


def _merge_way_nodes(graphs, kernel):
    clusters = {}
    nodes, ways = _get_way_nodes(graphs)
    # Find neighboring nodes
    half_kernel = kernel // 2
    for k, v in nodes.items():
        if len(v["ways"]) <= 2 or "cluster" in v:
            continue

        cluster_id = len(clusters)
        ngr_interxns = []
        for ox in range(-half_kernel, half_kernel + 1):
            for oy in range(-half_kernel, half_kernel + 1):
                ngr_nd_key = (k[0] + ox, k[1] + oy)
                if ngr_nd_key in nodes:
                    ngr_interxns.append(ngr_nd_key)

        if len(ngr_interxns) > 1:
            clusters[cluster_id] = ngr_interxns
            for ni in ngr_interxns:
                nodes[ni]["cluster"] = cluster_id

    # Replace way nodes
    for cv in clusters.values():
        mean_cord = (np.array(cv).mean(axis=0) + 0.5).astype(np.int32)
        for cn in cv:
            for w in nodes[cn]["ways"]:
                way_nodes = ways[w]["nodes"]
                for i, wn in enumerate(way_nodes):
                    if wn == cn:
                        way_nodes[i] = tuple(mean_cord)
            # Remove duplicated nodes in the way
            ways[w]["nodes"] = []
            for i in range(len(way_nodes)):
                if i == 0 or way_nodes[i] != way_nodes[i - 1]:
                    ways[w]["nodes"].append(way_nodes[i])

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for way in ways.values():
    #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    return [[v for v in ways.values() if len(v["nodes"]) >= 2]]


def _remove_short_orphan_ways(graphs, min_length):
    nodes, ways = _get_way_nodes(graphs)
    ways_to_remove = []
    for k, v in ways.items():
        if len(v["nodes"]) > min_length:
            continue

        end_nodes = [v["nodes"][0], v["nodes"][-1]]
        for en in end_nodes:
            if len(nodes[en]["ways"]) < 2:
                ways_to_remove.append(k)

    return [[v for k, v in ways.items() if k not in ways_to_remove]]


def _get_next_way_node(way_nodes, curr_node, n_step):
    N_DENSIFIED_NODES = 5
    assert way_nodes[0] == curr_node or way_nodes[-1] == curr_node
    # Densify the way nodes if only two nodes exist in the way
    if len(way_nodes) == 2:
        end_nodes = way_nodes
        way_nodes = [
            tuple(
                (
                    np.array(end_nodes[0]) * i / N_DENSIFIED_NODES
                    + np.array(end_nodes[1])
                    * (N_DENSIFIED_NODES - i)
                    / N_DENSIFIED_NODES
                ).astype(np.int32)
            )
            for i in range(N_DENSIFIED_NODES)
        ]

    if way_nodes[0] == curr_node:
        return way_nodes[n_step] if n_step < len(way_nodes) else None
    else:
        return way_nodes[-n_step - 1] if n_step < len(way_nodes) else None


def _get_intersection_point(pt0, pt1, pt2):
    if pt0[0] == pt1[0]:
        return (pt0[0], pt2[1])
    elif pt0[1] == pt1[1]:
        return (pt2[0], pt0[1])

    # The line equation of the line connecting pt0 and pt1
    k0 = (pt1[1] - pt0[1]) / (pt1[0] - pt0[0])
    b0 = pt0[1] - k0 * pt0[0]
    # The line perpendicular to the line connecting pt0 and pt1
    k1 = -1 / k0
    b1 = pt2[1] - k1 * pt2[0]
    # The intersection point
    x = (b1 - b0) / (k0 - k1)
    y = k0 * x + b0

    return (int(x + 0.5), int(y + 0.5))


def _insert_node_after(anchor_node, new_node, way_nodes):
    assert way_nodes[0] == anchor_node or way_nodes[-1] == anchor_node
    # No need to insert if the new node is the same as the anchor node
    if new_node == anchor_node:
        return way_nodes

    if way_nodes[0] == anchor_node:
        way_nodes.insert(0, new_node)
    else:
        way_nodes.append(new_node)

    return way_nodes


def _get_fixed_triangle_intersection(curr_node, connected_ways, max_angle):
    assert len(connected_ways) == 3

    choices = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    nodes_1st = [_get_next_way_node(cw["nodes"], curr_node, 1) for cw in connected_ways]
    nodes_2nd = [_get_next_way_node(cw["nodes"], curr_node, 2) for cw in connected_ways]
    # Skip if any of the nodes is None
    if any(n is None for n in nodes_1st) or any(n is None for n in nodes_2nd):
        return None

    nodes_1st = np.array([np.array(n) for n in nodes_1st])
    nodes_2nd = np.array([np.array(n) for n in nodes_2nd])
    best_choice = None
    best_choice_angle = max_angle
    for c in choices:
        idx0, idx1 = c[0], c[1]
        vec_0a = nodes_2nd[idx0] - nodes_1st[idx0]
        vec_0b = nodes_1st[idx0] - nodes_1st[idx1]
        angle0 = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(vec_0a, vec_0b)
                    / (np.linalg.norm(vec_0a) * np.linalg.norm(vec_0b)),
                    -1,
                    1,
                )
            )
        )
        vec_1a = nodes_2nd[idx1] - nodes_1st[idx1]
        vec_1b = nodes_1st[idx1] - nodes_1st[idx0]
        angle1 = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(vec_1a, vec_1b)
                    / (np.linalg.norm(vec_1a) * np.linalg.norm(vec_1b)),
                    -1,
                    1,
                )
            )
        )
        # Calculate the new intersection point
        if angle0 + angle1 < best_choice_angle:
            best_choice = c
            best_choice_angle = angle0 + angle1

    if best_choice is None:
        return None, None

    return connected_ways[best_choice[2]]["id"], _get_intersection_point(
        nodes_1st[best_choice[0]],
        nodes_1st[best_choice[1]],
        curr_node,
    )


def _fix_triangle_intersections(graphs, max_angle):
    nodes, ways = _get_way_nodes(graphs)
    for k, v in nodes.items():
        if len(v["ways"]) != 3:
            continue

        connected_ways = [ways[cw] for cw in v["ways"]]
        perpendicular_way, fixed_coord = _get_fixed_triangle_intersection(
            k, connected_ways, max_angle
        )
        # Skip if the intersection if the fixed condition is not met
        if fixed_coord is None:
            continue
        # Replace the fixed coordinates in the ways
        for cw in v["ways"]:
            way_nodes = ways[cw]["nodes"]
            if cw != perpendicular_way:
                for i, wn in enumerate(way_nodes):
                    if wn == k:
                        way_nodes[i] = fixed_coord
            else:
                ways[cw]["nodes"] = _insert_node_after(k, fixed_coord, way_nodes)

    return [[v for v in ways.values() if len(v["nodes"]) >= 2]]


def get_traffic_graphs(traffic_maps):
    traffic_graphs = {}
    for tk, tv in traffic_maps.items():
        traffic_graphs[tk] = {
            # "EDGE": _get_kpts_graph(tv["EDGE"], closed=True),
            "CNTR": _get_kpts_graph(tv["CNTR"], closed=False),
        }
        # Post-processing for the road centerlines
        traffic_graphs[tk]["CNTR"] = _remove_short_loops(traffic_graphs[tk]["CNTR"])
        traffic_graphs[tk]["CNTR"] = _merge_way_nodes(
            traffic_graphs[tk]["CNTR"], kernel=31
        )
        traffic_graphs[tk]["CNTR"] = _remove_short_orphan_ways(
            traffic_graphs[tk]["CNTR"], min_length=128
        )
        traffic_graphs[tk]["CNTR"] = _fix_triangle_intersections(
            traffic_graphs[tk]["CNTR"], max_angle=25
        )
        # Debug: Visualization
        # img = np.zeros((19600, 19600), np.uint8)
        # for way in traffic_graphs[tk]["CNTR"][0]:
        #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)

    return traffic_graphs


def get_road_widths(road_net, road_centers):
    dist_map = cv2.distanceTransform(road_net.astype(np.uint8), cv2.DIST_L2, 3)
    # TODO


def main(projection_dir, project_names):
    logging.info("Parsing Road Networks ...")
    road_networks = get_road_networks(projection_dir, project_names)
    logging.info("Parsing Traffic Maps ...")
    traffic_maps = get_traffic_maps(road_networks)
    # # Faster Debug
    # with open("output/traffic_maps.pkl", "rb") as f:
    #     # pickle.dump(traffic_maps, f)
    #     traffic_maps = pickle.load(f)
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
