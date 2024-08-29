# -*- coding: utf-8 -*-
#
# @File:   create_unreal_sequencer.py
# @Author: Haozhe Xie
# @Date:   2024-08-29 19:10:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-08-29 21:14:15
# @Email:  root@haozhexie.com

import csv
import unreal
import sys

CSV_FILE_PATH = "D:/Users/hzxie/Desktop/KeyFrames.csv"
SEQ_ASSET_PATH = "/Game/Sequences/TestSequence.TestSequence"

# Keyframe positions from CSV (frame number, location, rotation, scale)
keyframes = []
with open(CSV_FILE_PATH) as fp:
    reader = csv.DictReader(fp)
    for r in reader:
        r = {k: float(v) for k, v in r.items()}
        keyframes.append(
            (
                int(r["id"]),
                unreal.Vector(r["tx"], r["ty"], r["tz"]),
                unreal.Rotator(r["roll"], r["pitch"], r["yaw"]),
                unreal.Vector(1, 1, 1),
            )
        )

# Create a reference to the level sequence
sequence = unreal.load_asset(SEQ_ASSET_PATH)

if sequence is None:
    unreal.log_error(f"Sequence '{SEQ_ASSET_PATH}' not found in the level.")
    sys.exit()

tracks = sequence.get_master_tracks()
for binding in sequence.get_bindings():
    tracks.extend(binding.get_tracks())

if not tracks:
    unreal.log_error("No tracks found in the Cine Camera Actor.")
    sys.exit()

# Bind to the first track
track = next(t for t in tracks if t.get_name().find("MovieScene3DTransformTrack") != -1)

# Bind to the first section
sections = track.get_sections()
if not sections:
    unreal.log_warning("No sections found in the Cine Camera Actor.")
    section = track.add_section()
else:
    section = sections[0]

# Start from -30 frames to make the initial frames stable
keyframes.insert(0, keyframes[0])
keyframes[0][0] = -30
section.set_start_frame(-30)
section.set_end_frame(keyframes[-1][0])
# Get channels in this section
channels = section.get_all_channels()

# Add the keyframes to the section
for frame, location, rotation, scale in keyframes:
    # Add keyframes for location
    channels[0].add_key(unreal.FrameNumber(frame), location.x)
    channels[1].add_key(unreal.FrameNumber(frame), location.y)
    channels[2].add_key(unreal.FrameNumber(frame), location.z)
    # Add keyframes for rotation
    channels[3].add_key(unreal.FrameNumber(frame), rotation.pitch)
    channels[4].add_key(unreal.FrameNumber(frame), rotation.yaw)
    channels[5].add_key(unreal.FrameNumber(frame), rotation.roll)
    # Add keyframes for scale
    channels[6].add_key(unreal.FrameNumber(frame), scale.x)
    channels[7].add_key(unreal.FrameNumber(frame), scale.y)
    channels[8].add_key(unreal.FrameNumber(frame), scale.z)

# Save the level sequence
unreal.EditorAssetLibrary.save_asset(SEQ_ASSET_PATH)
unreal.log("Keyframes added successfully!")
