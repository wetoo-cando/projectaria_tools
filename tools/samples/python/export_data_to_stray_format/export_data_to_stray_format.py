# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List

import numpy as np

import rerun as rr
import rerun.blueprint as rrb

from PointsAndObservationsManager import (
    PointsAndObservationsManager,
    PointsDict,
    PointsIndexList,
    PointsUVsList,
)

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core import calibration
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline, ToTransform3D
from tqdm import tqdm
from TracksManager import MAX_TRACK_LENGTH, Tracks, TracksManager
from utils import OnlineRgbCameraHelper
import cv2
import random
import string
import shutil
from projectaria_tools.core.sophus import SE3

# MPS Semi Dense Point Visibility Demo
# - Show how to use Semi Dense Point Observations in conjunction of the VRS image frames
#
#
# Key learnings:
# 1. MPS Point cloud data consists of:
# - A global point cloud (3D points)
# - Point cloud observations (visibility information for camera_serial and timestamps)
# -> Global points and their observations are linked together by their unique 3D point ids
# => We are introducing the PointsAndObservationsManager class to enable easy retrieval of visible point per timestamp for each camera

# 2. How to connect MPS data to corresponding VRS image:
# MPS point cloud data observations is indexed by camera_serial and can be slam_left or slam_right
# MPS data is indexed by timestamp in microseconds
# VRS data is indexed by timestamp in nanoseconds

# 3. This code sample show how to keep a list of visible tracks at the current timestamp
# I.E point observations are accumulated and hashed by their global point unique ids
# - if a point is not visible on the current frame, the track is removed
# => We are introducing the TracksManager manager class to update and store the visible tracks


RERUN_JPEG_QUALITY = 75
# Create alias for the stream ids
LEFT_SLAM_STREAM_ID = StreamId("1201-1")
RIGHT_SLAM_STREAM_ID = StreamId("1201-2")
RGB_STREAM_ID = StreamId("214-1")

#
# Configure Data Loading
# # MPS output paths
# mps_folder_data = "../../../../data/mps_sample"
# vrs_file = os.path.join(mps_folder_data, "sample.vrs")
mps_folder_data = "../../../../../aria_data_from_glasses/mps_2025-05-16-DominikCmon_vrs"
vrs_file = os.path.join(mps_folder_data, "../2025-05-16-DominikCmon.vrs")


def get_world_to_odom_transform(device_pose_world):
    """
    Given device_pose_world (SE3), compute device_pose_odom such that:
    - Y_odom aligns to Z_world
    - X_odom aligns to -Y_device
    - Z_odom = X_odom x Y_odom (right-hand rule)
    - Origin of odom coincides with device origin
    Returns: world_to_odom_transform
    """
    T_wd = device_pose_world.to_matrix()  # Ensure it's a matrix
    R_wd = T_wd[:3, :3]  # 3x3 rotation matrix
    t_wd = T_wd[:3, 3]  # 3x1 translation vector
    print(f"R_wd: {R_wd}")
    print(f"t_wd: {t_wd}")

    # Device axes in world
    x_dev = R_wd[:, 0]
    y_dev = R_wd[:, 1]
    # z_dev = R_wd[:, 2]
    print(f"x_dev: {x_dev}")
    print(f"y_dev: {y_dev}")

    # Odom axes in world
    x_odom = -y_dev
    y_odom = np.array([0, 0, 1], dtype=float)
    z_odom = np.cross(x_odom, y_odom)

    # Normalize axes
    x_odom /= np.linalg.norm(x_odom)
    y_odom /= np.linalg.norm(y_odom)
    z_odom /= np.linalg.norm(z_odom)

    # Rotation from odom to world (columns are odom axes in world)
    R_wo = np.column_stack((x_odom, y_odom, z_odom))
    # Rotation from world to odom
    R_ow = R_wo.T

    # Translation from world to odom: origin coincides with device origin in world
    t_ow = -R_ow @ t_wd
    T_ow = np.eye(4)
    T_ow[:3, :3] = R_ow
    T_ow[:3, 3] = t_ow

    return T_ow


def undistort_image_and_calibration(
    input_image: np.ndarray,
    input_calib: calibration.CameraCalibration,
) -> [np.ndarray, calibration.CameraCalibration]:
    """
    Return the undistorted image and the updated camera calibration.
    """
    input_calib_width = input_calib.get_image_size()[0]
    input_calib_height = input_calib.get_image_size()[1]
    input_calib_focal = input_calib.get_focal_lengths()[0]
    if (
        # numpy array report matrix shape as (height, width)
        input_image.shape[0] != input_calib_height
        or input_image.shape[1] != input_calib_width
    ):
        raise ValueError(
            f"Input image shape {input_image.shape} does not match calibration {input_calib.get_image_size()}"
        )

    # Undistort the image
    pinhole = calibration.get_linear_camera_calibration(
        int(input_calib_width),
        int(input_calib_height),
        input_calib_focal,
        "pinhole",
        input_calib.get_transform_device_camera(),
    )
    updated_calib = pinhole
    output_image = calibration.distort_by_calibration(
        input_image, updated_calib, input_calib
    )

    return output_image, updated_calib

#
# Utility function
def display_tracks_and_points(
    uvs: PointsUVsList,
    uids: PointsIndexList,
    points_dict: PointsDict,  # Global Points indexed by uid
    tracks: Tracks,
    stream_id: StreamId,
    stream_label: str,
) -> None:
    """
    Display onto existing images:
     - 2D observations (uvs)
     - tracklets (the past trace of the tracked point)
     - 3D visible points (for SLAM left and right)
    """
    # Display the collected 2D point projections
    #
    rr.log(
        stream_label + "/observations",
        rr.Points2D(uvs, colors=[255, 255, 0], radii=1),
    )

    # Display tracklets
    #

    # Compile tracks as a list to display them as lines
    tracked_points = [tracks[track_id] for track_id in tracks]
    rr.log(
        stream_label + "/track",
        rr.LineStrips2D(tracked_points, radii=2),
    )

    # Collect visible 3D points and display them
    #
    if stream_id in [
        RGB_STREAM_ID
    ]:
        points = [points_dict[uid].position_world for uid in uids]
        rr.log(
            f"world/tracked_points_{stream_label}",
            rr.Points3D(
                points,
                radii=0.02,
                colors=(
                    [255, 0, 0] # if stream_id == LEFT_SLAM_STREAM_ID else [0, 0, 255]
                ),
            ),
        )


###
###

#
# Initialize the interface to read MPS data
mps_data_paths = MpsDataPathsProvider(mps_folder_data)
mps_data_provider = MpsDataProvider(mps_data_paths.get_data_paths())
# Check we have the required MPS data available
if not (
    mps_data_provider.has_closed_loop_poses
    and mps_data_provider.has_online_calibrations  # Required to have accurate RGB camera poses
    and mps_data_provider.has_semidense_point_cloud
    and mps_data_provider.has_semidense_observations
):
    print(
        "Missing required data for this demo (either closed loop trajectory, semi dense point cloud, semi dense observations, online calibration are missing)"
    )
    exit(1)

#
# Initialize the Semi Dense Points and observations manager
# This interface will provide us an easy way to retrieve visible points per camera stream and timestamp
points_and_observations_manager = PointsAndObservationsManager.from_mps_data_provider(
    mps_data_provider
)

#
# Configure a VRS data provider to get all the SLAM and RGB images
vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
device_calibration = vrs_data_provider.get_device_calibration()
deliver_option = vrs_data_provider.get_default_deliver_queued_options()
deliver_option.deactivate_stream_all()
deliver_option.activate_stream(LEFT_SLAM_STREAM_ID)
deliver_option.activate_stream(RIGHT_SLAM_STREAM_ID)
deliver_option.activate_stream(RGB_STREAM_ID)

#
# Load MPS trajectory data (and reduce the number of points for display)
trajectory_data = mps.read_closed_loop_trajectory(
    mps_data_paths.get_data_paths().slam.closed_loop_trajectory
)
device_trajectory = [
    it.transform_world_device.translation()[0] for it in trajectory_data
][0::80]


#
# Initialize Rerun and set the viewing layout (3D, RGB, VerticalStack(SlamLeft, SlamRight)):
rr.init("MPS SemiDensePoint Viewer", spawn=True)
my_blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin="world"),
        rrb.Vertical(
            rrb.Spatial2DView(origin="camera-rgb"),
            rrb.Spatial2DView(origin="depth_map"),
        )
    ),
    collapse_panels=True,
)
rr.send_blueprint(my_blueprint)


# Display device trajectory
rr.log(
    "world/device_trajectory",
    rr.LineStrips3D(device_trajectory, radii=0.008),
    static=True,
)

# # Display global point cloud
# points = [pt.position_world for pt in points_and_observations_manager.points.values()]
# rr.log("world/points", rr.Points3D(points, radii=0.005), static=True)
# del points  # Free memory (no longer needed)


#
# Log Aria Glasses outline
#
aria_glasses_point_outline = AriaGlassesOutline(device_calibration)
rr.log(
    "world/device/glasses_outline",
    rr.LineStrips3D([aria_glasses_point_outline]),
    static=True,
)

#
# Save tracks for each stream_id (the last X visible 2d projection coordinates for each visible track ID in the current frame)
aria_track_manager = TracksManager(max_track_length=MAX_TRACK_LENGTH)

#
# SemiDense point visibility information is done only for the SLAM cameras
# RGB point cloud visibility is not pre-computed, we estimate point visibilities from the SLAM ones
# - We are here store the Global point cloud unique ids that are visible in the current frame set (left and right)
# - so we can know later estimate which point is visible in the RGB frame
frame_set_uids: Dict[str, List[int]] = {}

#
# Display loop
# - going over the images
# - retrieve visible points
# - update tracks and display them
#
frame_counter = 0
export_dir = f"exported_data_stray_format"
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
os.makedirs(export_dir, exist_ok=True)
depth_export_dir = os.path.join(export_dir, "depth")
os.makedirs(depth_export_dir, exist_ok=True)
confidence_export_dir = os.path.join(export_dir, "confidence")
os.makedirs(confidence_export_dir, exist_ok=True)
rgb_export_dir = os.path.join(export_dir, "rgb")
os.makedirs(rgb_export_dir, exist_ok=True)
camera_matrix_path = os.path.join(export_dir, "camera_matrix.csv")
odometry_path = os.path.join(export_dir, "odometry.csv")
open(camera_matrix_path, "a").close()
open(odometry_path, "a").close()
# Write header to odometry.csv if file is empty
if os.stat(odometry_path).st_size == 0:
    with open(odometry_path, "w") as f:
        f.write("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
T_world_to_odom = None        

for data in tqdm(vrs_data_provider.deliver_queued_sensor_data(deliver_option)):
    device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
    rr.set_time_nanos("device_time", device_time_ns)

    # Display device pose
    closed_loop_pose = mps_data_provider.get_closed_loop_pose(device_time_ns)
    if closed_loop_pose:
        T_world_device = closed_loop_pose.transform_world_device
        rr.log(
            "world/device",
            ToTransform3D(T_world_device, False),
        )
    
    # Retrieve the stream label and current camera serial
    camera_serial = tracks = None
    stream_label = vrs_data_provider.get_label_from_stream_id(data.stream_id())
    if data.stream_id() == LEFT_SLAM_STREAM_ID:
        camera_serial = (
            vrs_data_provider.get_configuration(LEFT_SLAM_STREAM_ID)
            .image_configuration()
            .sensor_serial
        )
    elif data.stream_id() == RIGHT_SLAM_STREAM_ID:
        camera_serial = (
            vrs_data_provider.get_configuration(RIGHT_SLAM_STREAM_ID)
            .image_configuration()
            .sensor_serial
        )

    # If this is an image, display it
    if data.sensor_data_type() == SensorDataType.IMAGE:
        frame = data.image_data_and_record()[0].to_numpy_array()        

        # Collect and display "slam images" visible semi dense point 2D coordinates (uvs) and unique ids (uuids)
        if data.stream_id() in [
            LEFT_SLAM_STREAM_ID,
            RIGHT_SLAM_STREAM_ID,
        ]:
            uvs, uids = points_and_observations_manager.get_slam_observations(
                device_time_ns,
                camera_serial,
            )
            # Store the current visible global points uids for this view to propagate to the RGB view
            frame_set_uids[str(data.stream_id())] = uids

        # If we have accumulated SLAM image visibilities for both slam images, we can compute RGB visible points
        if len(frame_set_uids) == 2 and stream_label == "camera-rgb":
            
            # We will now estimate uvs and ids for the RGB view
            #
            # Collect visible 3D points for the SLAM images and see if they are "visible" in the RGB frame
            all_uids = set(frame_set_uids[str(LEFT_SLAM_STREAM_ID)]).union(
                frame_set_uids[str(RIGHT_SLAM_STREAM_ID)]
            )
            # Collect online camera calibration (used for point re-projection)
            camera_calibration, device_pose_world = OnlineRgbCameraHelper(
                vrs_data_provider, mps_data_provider, device_time_ns
            )

            undistorted_frame, undistorted_calib = undistort_image_and_calibration(frame, camera_calibration)

            undistorted_frame_upright = undistorted_frame.transpose(1, 0, 2) #[::-1, ...]  # Rotate 90 deg clockwise
            rr.log(stream_label, rr.Image(undistorted_frame_upright).compress(jpeg_quality=RERUN_JPEG_QUALITY))

            # Retrieve visible points and uids
            uvs, uids = points_and_observations_manager.get_rgb_observations(
                all_uids, undistorted_calib, device_pose_world
            )
            uvs = [(v, u) for (u, v) in uvs]

            #
            # Clean up the left/right slam camera uids cache for the next frame set iteration
            frame_set_uids = {}

            #
            # Update tracks and display (for the RGB view)
            rgb_stream_name = vrs_data_provider.get_label_from_stream_id(RGB_STREAM_ID)
            # aria_track_manager.update_tracks_and_remove_old_observations(
            #     rgb_stream_name, uvs, uids
            # )
            display_tracks_and_points(
                uvs,
                uids,
                points_and_observations_manager.points,
                aria_track_manager.get_track_for_camera_label(rgb_stream_name),
                RGB_STREAM_ID,
                rgb_stream_name,
            )

            if len(uids) > 0:
                rgb_shape = frame.shape[:2]
                _, depth_map = points_and_observations_manager.construct_depth_map_from_3Dpoints_visible_in_rgb_img(
                    rgb_shape, all_uids, undistorted_calib, device_pose_world
                )

                # # Display the depth map
                # Check for valid max value
                max_val = np.nanmax(depth_map)
                if max_val == 0 or np.isnan(max_val):
                    raise ValueError("depth_map contains only zeros or NaNs.")
                # Replace NaN/inf with 0 before scaling
                safe_depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
                # Generate uint8 depth map
                depth_map_uint8 = np.clip(safe_depth_map / max_val * 255, 0, 255).astype(np.uint8)
                depth_map_uint8 = depth_map_uint8.T  # Interchange height and width
                # Invert the depth: closer points get higher values, farther get lower, except zeros and NaNs
                inverted_depth_map = np.where(depth_map_uint8 > 0, 255 - depth_map_uint8, depth_map_uint8)
                depth_map_uint8 = inverted_depth_map
                rr.log(
                    "depth_map",
                    rr.Image(depth_map_uint8).compress(jpeg_quality=RERUN_JPEG_QUALITY),
                )
                
                # # Store the depth map as a PNG file
                # Convert depth_map (float16, meters, with NaNs) to uint16 (millimeters), NaNs set to zero
                depth_map_mm = np.nan_to_num(depth_map, nan=0.0) * 1000.0  # Convert to millimeters, NaNs to 0
                depth_map_uint16 = np.clip(depth_map_mm, 0, 65535).astype(np.uint16)
                print(
                    f" [{frame_counter:6d}] Depth map uint16 stats: min={np.min(depth_map_uint16)}, max={np.max(depth_map_uint16)}, mean={np.mean(depth_map_uint16):.2f}, std={np.std(depth_map_uint16):.2f}, shape={depth_map_uint16.shape}"
                )
                export_path_depth = os.path.join(depth_export_dir, f"{frame_counter:06d}.png")                
                cv2.imwrite(export_path_depth, depth_map_uint16)

                # Create a confidence map with the same shape as depth_map_uint8, filled with zeros (uint8)
                confidence_map = 3 * np.ones_like(depth_map_uint8, dtype=np.uint8)
                export_path_confidence = os.path.join(confidence_export_dir, f"{frame_counter:06d}.png")
                cv2.imwrite(export_path_confidence, confidence_map)

                export_path_rgb = os.path.join(rgb_export_dir, f"{frame_counter:06d}.png")
                # Convert undistorted_frame from RGB to BGR for OpenCV before saving
                undistorted_frame_bgr = cv2.cvtColor(undistorted_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(export_path_rgb, undistorted_frame_bgr)                

                # Export undistorted_calib to camera_matrix.csv
                fx, fy = undistorted_calib.get_focal_lengths()
                cx, cy = undistorted_calib.get_principal_point()
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]
                ])
                # Save as a 3x3 matrix (overwrite each time)
                with open(camera_matrix_path, "w") as f:
                    np.savetxt(f, camera_matrix, delimiter=",", fmt="%.8f")

                # Export odometry (camera pose) to odometry.csv
                if T_world_to_odom is None:
                    T_world_to_odom = get_world_to_odom_transform(device_pose_world)
                device_pose_odom = SE3.from_matrix(T_world_to_odom @ device_pose_world.to_matrix())
                quaternion_and_translation = device_pose_odom.to_quat_and_translation()
                w, x, y, z, tx, ty, tz = quaternion_and_translation[0]                
                t = (tx, ty, tz)
                q = (x, y, z, w)  # OpenCV/ROS convention: (x, y, z, w)
                with open(odometry_path, "a") as f:
                    timestamp_sec = device_time_ns / 1e9
                    f.write(f"{timestamp_sec:.9f},{frame_counter:06d},{t[0]:.9f},{t[1]:.9f},{t[2]:.9f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},{q[3]:.9f}\n")

                frame_counter += 1
                if frame_counter >900:
                    break


# Export all RGB images as a video (rgb.mp4) after processing the last frame
if frame_counter == len(os.listdir(rgb_export_dir)):
    rgb_images = []
    for i in range(frame_counter):
        img_path = os.path.join(rgb_export_dir, f"{i:06d}.png")
        img = cv2.imread(img_path)
        if img is not None:
            rgb_images.append(img)
    print(f"Number of RGB images: {len(rgb_images)} being written to mp4")
    if rgb_images:
        height, width, _ = rgb_images[0].shape
        video_path = os.path.join(export_dir, "rgb.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for img in rgb_images:
            out.write(img)
        out.release()