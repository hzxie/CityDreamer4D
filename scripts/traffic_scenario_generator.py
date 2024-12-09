# -*- coding: utf-8 -*-
#
# @File:   traffic_scenario_generator.py
# @Author: Haozhe Xie
# @Date:   2024-11-02 15:17:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-12-09 11:12:37
# @Email:  root@haozhexie.com

import argparse
import cv2
import json
import logging
import math
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

    CFG = {"LANE_WIDTH": 14, "CENTERLINE_WIDTH": 10}

    return CFG[key] if key in CFG else cfg.DATASETS.CITY_SAMPLE[key]


def get_road_networks(projection_dir, project_names):
    classes = get_cfg_values("CLASSES")
    projections = {
        pn: np.array(Image.open(os.path.join(projection_dir, "%s_INS_BEV.png" % pn)))
        for pn in project_names
    }
    for k, v in projections.items():
        assert k in ["REST", "FREEWAY"]
        v[v != classes["ROAD"]] = classes["NULL"]
        v = scipy.ndimage.gaussian_filter(v * 255, sigma=7)
        projections[k] = (v >= 128).astype(np.uint8)

    # Debug: Visualization
    # import utils.helpers
    # utils.helpers.get_seg_map(projections["FREEWAY"]).save("output/test.png")
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

    return [{"nodes": way} for way in ways]


def _simplify_ways(_ways, tolerance=12):
    ways = []
    for way in tqdm(_ways, desc="Simplifying ways", leave=False):
        way_nodes = way["nodes"]
        is_closed = way_nodes[0] == way_nodes[-1]
        if len(set(way_nodes)) == 1:
            continue
        elif is_closed and len(set(way_nodes)) <= 2:
            continue

        if is_closed:
            shape = shapely.simplify(shapely.Polygon(way_nodes), tolerance=tolerance)
            x, y = shape.exterior.coords.xy
        else:
            shape = shapely.simplify(shapely.LineString(way_nodes), tolerance=tolerance)
            x, y = shape.coords.xy

        ways.append(
            {
                "nodes": np.array([np.array(x), np.array(y)]).astype(np.int32).T,
            }
        )
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

    ways = []
    for i in tqdm(range(1, n_conn), desc="Pasring connected graphs", leave=False):
        _kpts_map = kpts_map * (conn_map == i)
        kp_ys, kp_xs = np.where(_kpts_map != 0)  # ([ys], [xs])
        _nodes = _get_nodes(_kpts_map, kp_xs, kp_ys, closed)
        _ways = _get_ways(_nodes, closed)
        _ways = _simplify_ways(_ways)
        ways.extend(_ways)

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for way in ways:
    #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    return ways


def _get_way_nodes(_ways):
    nodes = {}
    ways = {}
    # Organzing connectivity
    for way in _ways:
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
    if len(ways_to_connect) > 3:
        logging.warning("Too many ways to connect: %s" % ways_to_connect)
        return ways

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


def _remove_short_loops(ways):
    nodes, ways = _get_way_nodes(ways)
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
    return list(ways.values())


def _get_mean_intersection_coord(intersections, **_):
    return (np.array(intersections).mean(axis=0) + 0.5).astype(np.int32)


def _get_nearby_nodes(curr_node, kernel, nodes):
    ngr_nodes = []
    for ox in range(-kernel, kernel + 1):
        for oy in range(-kernel, kernel + 1):
            ngr_nd_key = (curr_node[0] + ox, curr_node[1] + oy)
            if ngr_nd_key in nodes:
                ngr_nodes.append(ngr_nd_key)

    return ngr_nodes


def _remove_duplicated_nodes(way_nodes):
    _way_nodes = [way_nodes[0]]
    for i in range(1, len(way_nodes)):
        if way_nodes[i] != way_nodes[i - 1]:
            _way_nodes.append(way_nodes[i])

    return _way_nodes


def _merge_nearby_intersections(ways, kernel, new_cord_callback):
    clusters = {}
    nodes, ways = _get_way_nodes(ways)
    # Find neighboring nodes
    half_kernel = kernel // 2
    for k, v in nodes.items():
        if len(v["ways"]) <= 2 or "cluster" in v:
            continue

        cluster_id = len(clusters)
        ngr_interxns = _get_nearby_nodes(k, half_kernel, nodes)
        if len(ngr_interxns) > 1:
            clusters[cluster_id] = ngr_interxns
            for ni in ngr_interxns:
                nodes[ni]["cluster"] = cluster_id

    # Replace way nodes
    for cv in clusters.values():
        mean_cord = new_cord_callback(cv, nodes=nodes, ways=ways)
        for cn in cv:
            for w in nodes[cn]["ways"]:
                way_nodes = ways[w]["nodes"]
                for i, wn in enumerate(way_nodes):
                    if wn == cn:
                        way_nodes[i] = tuple(mean_cord)
            # Remove duplicated nodes in the way
            ways[w]["nodes"] = _remove_duplicated_nodes(way_nodes)

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for way in ways.values():
    #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    return [v for v in ways.values() if len(v["nodes"]) >= 2]


def _remove_short_orphan_ways(ways, min_length):
    nodes, ways = _get_way_nodes(ways)
    ways_to_remove = []
    for k, v in ways.items():
        if _get_way_length(v["nodes"]) > min_length:
            continue

        end_nodes = [v["nodes"][0], v["nodes"][-1]]
        for en in end_nodes:
            if len(nodes[en]["ways"]) < 2:
                ways_to_remove.append(k)

    return [v for k, v in ways.items() if k not in ways_to_remove]


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


def _get_line_equation(pt0, pt1):
    if pt0[0] == pt1[0]:
        return None, pt0[0]

    k = (pt1[1] - pt0[1]) / (pt1[0] - pt0[0])
    b = pt0[1] - k * pt0[0]
    return k, b


def _get_intersection_point(line0, line1):
    k0, b0 = line0
    k1, b1 = line1
    if k0 == k1:
        return None
    elif k0 is None:
        return (b0, k1 * b0 + b1)
    elif k1 is None:
        return (b1, k0 * b1 + b0)

    x = (b1 - b0) / (k0 - k1)
    y = k0 * x + b0
    return (int(x + 0.5), int(y + 0.5))


def _get_perpendicular_intersection_point(pt0, pt1, pt2):
    # The line equation of the line connecting pt0 and pt1
    k0, b0 = _get_line_equation(pt0, pt1)
    # The line perpendicular to the line connecting pt0 and pt1
    k1 = 0 if k0 is None else (-1 / k0 if k0 != 0 else None)
    b1 = pt2[1] - k1 * pt2[0] if k1 is not None else pt2[0]
    # The intersection point
    return _get_intersection_point((k0, b0), (k1, b1))


def _get_vector(node0, node1):
    return np.array(node0) - np.array(node1)


def _get_vector_angle(vec0, vec1):
    angle = np.degrees(
        np.arccos(
            np.clip(
                np.dot(vec0, vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1)),
                -1,
                1,
            )
        )
    )
    return angle


def _get_fixed_triangle_intersection(curr_node, connected_ways, max_angle):
    assert len(connected_ways) == 3
    choices = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    nodes_1st = [_get_next_way_node(cw["nodes"], curr_node, 1) for cw in connected_ways]
    nodes_2nd = [_get_next_way_node(cw["nodes"], curr_node, 2) for cw in connected_ways]
    # Skip if any of the nodes is None
    if any(n is None for n in nodes_1st) or any(n is None for n in nodes_2nd):
        return None

    best_choice = None
    best_choice_angle = max_angle
    for c in choices:
        idx0, idx1 = c[0], c[1]
        angle0 = _get_vector_angle(
            _get_vector(nodes_2nd[idx0], nodes_1st[idx0]),
            _get_vector(nodes_1st[idx0], nodes_1st[idx1]),
        )
        angle1 = _get_vector_angle(
            _get_vector(nodes_2nd[idx1], nodes_1st[idx1]),
            _get_vector(nodes_1st[idx1], nodes_1st[idx0]),
        )
        # Calculate the new intersection point
        if angle0 + angle1 < best_choice_angle:
            best_choice = c
            best_choice_angle = angle0 + angle1

    if best_choice is None:
        return None

    return _get_perpendicular_intersection_point(
        nodes_1st[best_choice[0]],
        nodes_1st[best_choice[1]],
        curr_node,
    )


def _fix_triangle_intersections(ways, max_angle):
    nodes, ways = _get_way_nodes(ways)
    for k, v in nodes.items():
        if len(v["ways"]) != 3:
            continue

        connected_ways = [ways[cw] for cw in v["ways"]]
        fixed_coord = _get_fixed_triangle_intersection(k, connected_ways, max_angle)
        # Skip if the intersection if the fixed condition is not met
        if fixed_coord is None:
            continue
        # Replace the fixed coordinates in the ways
        for cw in v["ways"]:
            way_nodes = ways[cw]["nodes"]
            for i, wn in enumerate(way_nodes):
                if wn == k:
                    way_nodes[i] = fixed_coord

    return [v for v in ways.values() if len(v["nodes"]) >= 2]


def _manually_fix_intersections(ways, intersections):
    nodes, ways = _get_way_nodes(ways)
    interxns = {k: v for k, v in nodes.items() if len(v["ways"]) > 2}
    for interxn in intersections:
        ngr_interxns = _get_nearby_nodes(interxn["node"], interxn["kernel"], interxns)
        if "include" in interxn:
            ngr_interxns.extend([tuple(n) for n in interxn["include"]])

        for ni in ngr_interxns:
            if ni not in nodes:
                logging.warning("The node %s is not found." % (ni,))
                continue
            for w in nodes[ni]["ways"]:
                way_nodes = ways[w]["nodes"]
                for i, wn in enumerate(way_nodes):
                    interxn_dist = np.linalg.norm(
                        np.array(way_nodes[i]) - np.array(interxn["node"])
                    )
                    if wn == ni or interxn_dist < interxn["kernel"]:
                        way_nodes[i] = tuple(interxn["node"])
            # Remove duplicated nodes in the way
            ways[w]["nodes"] = _remove_duplicated_nodes(way_nodes)

    return [v for v in ways.values() if len(v["nodes"]) >= 2]


def is_intersection_within_segment(interxn, seg0, seg1):
    min_x0 = min(seg0[0][0], seg0[1][0])
    max_x0 = max(seg0[0][0], seg0[1][0])
    min_y0 = min(seg0[0][1], seg0[1][1])
    max_y0 = max(seg0[0][1], seg0[1][1])
    min_x1 = min(seg1[0][0], seg1[1][0])
    max_x1 = max(seg1[0][0], seg1[1][0])
    min_y1 = min(seg1[0][1], seg1[1][1])
    max_y1 = max(seg1[0][1], seg1[1][1])
    return (
        interxn[0] >= min_x0
        and interxn[0] <= max_x0
        and interxn[1] >= min_y0
        and interxn[1] <= max_y0
        and interxn[0] >= min_x1
        and interxn[0] <= max_x1
        and interxn[1] >= min_y1
        and interxn[1] <= max_y1
    )


def _get_freeway_intersections(freeway_entry, way_nodes):
    entry_line = _get_line_equation(freeway_entry[0], freeway_entry[1])
    for i in range(1, len(way_nodes)):
        interxn = _get_intersection_point(
            _get_line_equation(way_nodes[i - 1], way_nodes[i]), entry_line
        )
        if interxn is None:
            continue
        # Check if the intersection is within the freeway_entry line segment
        if is_intersection_within_segment(
            interxn, freeway_entry, (way_nodes[i - 1], way_nodes[i])
        ):
            return interxn, way_nodes[i - 1]

    return None, None


def _get_attached_way(freeway_entry, road_interxns, road_ways):
    min_dist = float("inf")
    min_dist_interxn = None
    # Find the nearest intersection
    for ri in road_interxns.keys():
        dist = np.linalg.norm(np.array(ri) - np.array(freeway_entry[0]))
        if dist < min_dist:
            min_dist = dist
            min_dist_interxn = ri

    # Find the attached way
    attached_way = None
    candidate_ways = road_interxns[min_dist_interxn]["ways"]
    for cw in candidate_ways:
        assert (
            road_ways[cw]["nodes"][0] == min_dist_interxn
            or road_ways[cw]["nodes"][-1] == min_dist_interxn
        )
        if road_ways[cw]["nodes"][-1] == min_dist_interxn:
            road_ways[cw]["nodes"] = list(reversed(road_ways[cw]["nodes"]))

        freeway_interxn, anchor_way_node = _get_freeway_intersections(
            freeway_entry, road_ways[cw]["nodes"]
        )
        if freeway_interxn is not None:
            attached_way = cw
            return attached_way, freeway_interxn, anchor_way_node

    return None, None, None


def _insert_node_after(anchor_node, new_node, way_nodes):
    assert anchor_node in way_nodes
    # No need to insert if the new node is the same as the anchor node
    if new_node == anchor_node:
        return way_nodes

    anchor_idx = way_nodes.index(anchor_node)
    way_nodes.insert(anchor_idx + 1, new_node)
    return way_nodes


def _attach_freeways_and_roads(freeways, roads):
    free_nodes, free_ways = _get_way_nodes(freeways)
    road_nodes, road_ways = _get_way_nodes(roads)
    road_interxns = {k: v for k, v in road_nodes.items() if len(v["ways"]) > 2}

    freeway_entries = [k for k, v in free_nodes.items() if len(v["ways"]) == 1]
    for fe in freeway_entries:
        freeway_id = free_nodes[fe]["ways"][0]
        entry_next = _get_next_way_node(free_ways[freeway_id]["nodes"], fe, 1)
        attached_way, freeway_interxn, anchor_way_node = _get_attached_way(
            (fe, entry_next), road_interxns, road_ways
        )
        if attached_way is None:
            logging.warning("No attached way found for freeway entry: %s" % (fe,))
            continue
        # Add intersection to the attached way nodes
        road_ways[attached_way]["nodes"] = _insert_node_after(
            anchor_way_node, freeway_interxn, road_ways[attached_way]["nodes"]
        )
        # Change the entry node to the intersection
        assert (
            free_ways[freeway_id]["nodes"][0] == fe
            or free_ways[freeway_id]["nodes"][-1] == fe
        )
        if free_ways[freeway_id]["nodes"][-1] == fe:
            free_ways[freeway_id]["nodes"] = list(
                reversed(free_ways[freeway_id]["nodes"])
            )

        free_ways[freeway_id]["nodes"][0] = freeway_interxn

    return [v for v in free_ways.values() if len(v["nodes"]) >= 2], [
        v for v in road_ways.values() if len(v["nodes"]) >= 2
    ]


def get_traffic_graphs(traffic_maps, manual_fixer):
    traffic_graphs = {}
    for tk, tv in traffic_maps.items():
        traffic_graphs[tk] = {
            "EDGE": _get_kpts_graph(tv["EDGE"], closed=True),
            "CNTR": _get_kpts_graph(tv["CNTR"], closed=False),
        }
        # Post-processing for the road centerlines
        traffic_graphs[tk]["CNTR"] = _remove_short_loops(traffic_graphs[tk]["CNTR"])
        traffic_graphs[tk]["CNTR"] = _merge_nearby_intersections(
            traffic_graphs[tk]["CNTR"],
            kernel=31,
            new_cord_callback=_get_mean_intersection_coord,
        )
        traffic_graphs[tk]["CNTR"] = _remove_short_orphan_ways(
            traffic_graphs[tk]["CNTR"], min_length=128
        )
        traffic_graphs[tk]["CNTR"] = _fix_triangle_intersections(
            traffic_graphs[tk]["CNTR"], max_angle=10
        )
        traffic_graphs[tk]["CNTR"] = _fix_triangle_intersections(
            traffic_graphs[tk]["CNTR"], max_angle=25
        )
        if manual_fixer is not None and tk in manual_fixer:
            traffic_graphs[tk]["CNTR"] = _manually_fix_intersections(
                traffic_graphs[tk]["CNTR"], manual_fixer[tk]["intersections"]
            )
        traffic_graphs[tk]["CNTR"] = _simplify_ways(traffic_graphs[tk]["CNTR"])

    # Attach freeways to roads
    traffic_graphs["FREEWAY"]["CNTR"], traffic_graphs["REST"]["CNTR"] = (
        _attach_freeways_and_roads(
            traffic_graphs["FREEWAY"]["CNTR"], traffic_graphs["REST"]["CNTR"]
        )
    )

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for way in traffic_graphs["REST"]["CNTR"]:
    #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    # for way in traffic_graphs["FREEWAY"]["CNTR"]:
    #     img = cv2.polylines(img, [np.array(way["nodes"])], False, 255, 1)
    return traffic_graphs


def _get_way_widths(road_network, road_centers, lane_width, centerline_width):
    dist_map = cv2.distanceTransform(road_network.astype(np.uint8), cv2.DIST_L2, 3)
    for rc in road_centers:
        min_width = float("inf")
        n_nodes = len(rc["nodes"])
        for i in range(1, n_nodes):
            mid_pt = (np.array(rc["nodes"][i - 1]) + np.array(rc["nodes"][i])) / 2
            mid_pt = (int(mid_pt[0] + 0.5), int(mid_pt[1] + 0.5))
            width = dist_map[mid_pt[1], mid_pt[0]]
            if width < min_width:
                min_width = width

        rc["width"] = int(min_width + 0.5)
        rc["n_lanes"] = math.floor((rc["width"] - centerline_width) / lane_width)

    return road_centers


def _get_next_node_along_path(way_nodes, dist):
    # Step 1: Iterate through the points to calculate distances
    accumulated_distance = 0
    for i in range(1, len(way_nodes)):
        x1, y1 = way_nodes[i - 1]
        x2, y2 = way_nodes[i]
        segment_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if accumulated_distance + segment_distance >= dist:
            # Step 2: Find the remaining distance to cover in this segment
            remaining_distance = dist - accumulated_distance
            # Step 3: Interpolate to find the new point
            ratio = remaining_distance / segment_distance
            x_new = x1 + ratio * (x2 - x1)
            y_new = y1 + ratio * (y2 - y1)
            return int(x_new + 0.5), int(y_new + 0.5)

        # Accumulate the distance and move to the next segment
        accumulated_distance += segment_distance

    return None, None


def _get_shortened_ways(road_centers):
    # Organizing the intersections
    interxns = {}
    for rc in road_centers:
        for n in [rc["nodes"][0], rc["nodes"][-1]]:
            if n not in interxns:
                interxns[n] = {"ways": [], "width": 0}
            interxns[n]["ways"].append(rc["id"])
            interxns[n]["width"] = max(interxns[n]["width"], rc["width"])

    interxns = {k: v for k, v in interxns.items() if len(v) > 1}

    # Shorten the ways
    for k, v in interxns.items():
        for w in v["ways"]:
            way_nodes = road_centers[w]["nodes"]
            # Remove duplicated nodes in the way
            # way_nodes = _remove_duplicated_nodes(way_nodes)
            assert way_nodes[0] == k or way_nodes[-1] == k

            if way_nodes[0] == k:
                _node = _get_next_node_along_path(way_nodes, v["width"])
                way_nodes.insert(1, _node)
            else:
                _node = _get_next_node_along_path(way_nodes[::-1], v["width"])
                way_nodes.insert(-1, _node)

    return road_centers


def _is_nodes_reversed(nodes):
    if len(nodes) < 2:
        return False

    first_node = nodes[0]
    last_node = nodes[-1]
    if first_node[0] < last_node[0]:
        return False
    elif first_node[0] > last_node[0]:
        return True

    return first_node[1] < last_node[1]


def _get_traffic_lanes(road_centers, lane_width):
    lanes = []
    for rc in road_centers:
        nodes = rc["nodes"][1:-1]
        nodes = nodes[::-1] if _is_nodes_reversed(nodes) else nodes
        n_nodes = len(nodes)

        vectors = []
        for i in range(1, n_nodes):
            vec = _get_vector(nodes[i], nodes[i - 1])
            vec = vec / np.linalg.norm(vec)
            vec = np.array([-vec[1], vec[0]])
            vectors.append(vec)

        for i in range(1, rc["n_lanes"] + 1):
            _fwd_lane, _bwd_lane = [], []
            for n in nodes:
                _fwd_lane.append(
                    tuple((np.array(n) - vec * lane_width * i).astype(np.int32))
                )
                _bwd_lane.append(
                    tuple((np.array(n) + vec * lane_width * i).astype(np.int32))
                )

            lanes.append({"way": rc["id"], "nodes": _fwd_lane[::-1], "dir": "F"})
            lanes.append({"way": rc["id"], "nodes": _bwd_lane, "dir": "B"})

    # Debug: Visualization
    # img = np.zeros((19600, 19600), np.uint8)
    # for lane in lanes:
    #     img = cv2.polylines(img, [np.array(lane["nodes"])], False, 255, 1)
    return lanes


def get_traffic_lanes(road_networks, traffic_graphs, lane_width, centerline_width):
    for tk, tv in traffic_graphs.items():
        tv["CNTR"] = _get_way_widths(
            road_networks[tk], tv["CNTR"], lane_width, centerline_width
        )
        tv["CNTR"] = _get_shortened_ways(tv["CNTR"])
        tv["LANE"] = _get_traffic_lanes(tv["CNTR"], lane_width)
        # TODO
        # tv["LANE"] = _connect_intersection_lanes(tv["LANE"])
        # tv["LINE"] = None

    return traffic_graphs


def _get_heading(cx, cy, way_nodes):
    for i in range(1, len(way_nodes)):
        prev_node = way_nodes[i - 1]
        next_node = way_nodes[i]
        if (
            cx >= min(prev_node[0], next_node[0])
            and cx <= max(prev_node[0], next_node[0])
            and cy >= min(prev_node[1], next_node[1])
            and cy <= max(prev_node[1], next_node[1])
        ):
            delta_x = next_node[0] - prev_node[0]
            delta_y = next_node[1] - prev_node[1]
            return math.degrees(math.atan2(delta_y, delta_x))

    # assert False, "The point is not on the way."
    return None


def _get_vehicles_along_lane(traffic_lane):
    MIN_LANE_LENGTH = 100

    vehicles = []
    lane_length = _get_way_length(traffic_lane["nodes"])
    if lane_length < MIN_LANE_LENGTH:
        return vehicles

    while True:
        offset = MIN_LANE_LENGTH * len(vehicles) + np.random.randint(-25, 25)
        cx, cy = _get_next_node_along_path(traffic_lane["nodes"], offset)
        if cx is None and cy is None:
            break

        heading = _get_heading(cx, cy, traffic_lane["nodes"])
        if heading is None:
            continue

        vehicles.append({"cx": cx, "cy": cy, "heading": heading})

    # Random dropout
    return [v for v in vehicles if np.random.rand() > 0.5]


def _get_vehicles_along_lanes(traffic_lanes):
    tracks = []
    for tl in traffic_lanes:
        if "dir" not in tl:
            # DO NOT put vehicles on the intersections
            continue

        tracks.extend(_get_vehicles_along_lane(tl))

    return tracks


def generate_init_scenario(traffic_lanes):
    scenario = {"timestamps_seconds": 0}
    for tv in traffic_lanes.values():
        scenario["TRACK"] = _get_vehicles_along_lanes(tv)
        # TODO
        # scenario["TLIGHT"] = None

    return scenario


def main(projection_dir, manual_fix_file, project_names, n_steps):
    logging.info("Parsing Road Networks ...")
    road_networks = get_road_networks(projection_dir, project_names)
    logging.info("Parsing Traffic Maps ...")
    traffic_maps = get_traffic_maps(road_networks)
    # # Faster Debug
    # with open("output/traffic_maps.pkl", "rb") as f:
    #     # pickle.dump(traffic_maps, f)
    #     traffic_maps = pickle.load(f)

    manual_fixer = None
    if os.path.exists(manual_fix_file):
        manual_fixer = json.loads(open(manual_fix_file, "r").read())

    logging.info("Parsing Traffic Graphs ...")
    traffic_graphs = get_traffic_graphs(traffic_maps, manual_fixer)
    # # Faster Debug
    # with open("output/traffic_graphs.pkl", "rb") as f:
    #     # pickle.dump(traffic_graphs, f)
    #     traffic_graphs = pickle.load(f)

    traffic_graphs = get_traffic_lanes(
        road_networks,
        traffic_graphs,
        get_cfg_values("LANE_WIDTH"),
        get_cfg_values("CENTERLINE_WIDTH"),
    )
    # # Faster Debug
    # with open("output/traffic_lanes.pkl", "rb") as f:
    #     # pickle.dump(traffic_graphs, f)
    #     traffic_graphs = pickle.load(f)

    logging.info("Generating Traffic Scenarios ...")
    scenarios = []
    scenarios.append(
        generate_init_scenario({k: v["LANE"] for k, v in traffic_graphs.items()})
    )
    # TODO: Generate scenarios in the next steps
    # for i in range(n_steps):
    #     scenarios.generate_next_scenario(scenarios[-1])

    # logging.info("Convert Traffic Scenarios to BEV Maps ...")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="The Traffic Scenario Generator")
    parser.add_argument(
        "--projection_dir",
        default=os.path.join(PROJECT_HOME, "data", "city-sample", "%s", "Projections"),
    )
    parser.add_argument(
        "--manual_fix_file",
        default=os.path.join(
            PROJECT_HOME, "data", "city-sample", "%s", "TrafficFix.json"
        ),
    )
    parser.add_argument("--city", default="City01")
    parser.add_argument("--project_names", default="REST, FREEWAY")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    main(
        args.projection_dir % (args.city),
        args.manual_fix_file % (args.city),
        [pn.strip() for pn in args.project_names.split(",")],
        args.steps,
    )
