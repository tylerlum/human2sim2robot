# Human Demonstration

The goal of this section is to collect and process human video demonstrations of a task.

## Collect Raw Data

For this step, you collect data in `data/human_demo_raw_data/`.

Set the task name and path to the object .obj file:
```
export TASK_NAME=my_task_name
```

First, run the following to start the ZED camera:

```
roslaunch zed_wrapper zed.launch
```

Then check that the desired topics are published for RGB, depth, and camera info:
```
rostopic list
```

Then, run the following to collect the data:

```
python human2sim2robot/human_demo/collect_rgbd_demo.py \
--output_dir data/human_demo_raw_data/$TASK_NAME \
--rgb_topic /zed/zed_node/rgb/image_rect_color \
--depth_topic /zed/zed_node/depth/depth_registered \
--camera_info_topic /zed/zed_node/rgb/camera_info
```

This creates the following:

```
data/human_demo_raw_data/$TASK_NAME/
  ├── rgb/
  |   ├── 00000.png
  |   ├── 00001.png
  |   └── ...
  ├── depth/
  |   ├── 00000.png
  |   ├── 00001.png
  |   └── ...
  └── cam_K.txt
```

WARNING: Be careful about the depth images. The conventions of depth images are not standardized, so you may need to convert them to a standard format.

* When the depth image doesn't see anything at a given pixel, it is often set to 0.0 or NaN. Make sure to convert these to 0.0.
* Depth image measurements are often in meters or millimeters. Make sure to convert them to meters.

Sanity check data by visualizing a 3D point cloud:
```
python human2sim2robot/human_demo/visualize_point_cloud.py \
--rgb_dir data/human_demo_raw_data/$TASK_NAME/rgb/ \
--depth_dir data/human_demo_raw_data/$TASK_NAME/depth/ \
--cam-intrinsics-path data/human_demo_raw_data/$TASK_NAME/cam_K.txt \
--idx 0
```

## Creating Object and Hand Segmentation Masks

The goal of this section is to create segmentation masks of the object and the hand using Segment Anything Model 2 (SAM 2). This requires two human prompts: one for selecting a point on the object/hand (positive prompt) and one for selecting a point NOT on the object/hand (negative prompt). Make sure that you have installed SAM 2 (instructions [here](../../docs/installation.md)). Then run the following:

```
python thirdparty/segment-anything-2-real-time/video_sam2.py \
--input_dir data/human_demo_raw_data/$TASK_NAME/rgb/ \
--output_dir data/human_demo_raw_data/$TASK_NAME/masks/

python thirdparty/segment-anything-2-real-time/video_sam2.py \
--input_dir data/human_demo_raw_data/$TASK_NAME/rgb/ \
--output_dir data/human_demo_raw_data/$TASK_NAME/hand_masks/ \
--use_negative_prompt
```

This creates the following:
```
data/human_demo_raw_data/$TASK_NAME/
  ├── masks/
  |   ├── 00000.png
  |   ├── 00001.png
  |   └── ...
  └── hand_masks/
      ├── 00000.png
      ├── 00001.png
      └── ...
```

## Predicting Object Pose Trajectory

The goal of this section is to predict the object pose trajectory using FoundationPose. Make sure that you have installed FoundationPose (instructions [here](../../docs/installation.md)). 

Before running FoundationPose, you need to have a 3D mesh of the object. See [here](../real_to_sim/README.md) for more information.

Enter the FoundationPose docker container and make sure it is able to access to `data` folder. Then run the following:

```
export OBJ_PATH=assets/my_object_processed/my_object.obj
```

```
cd <FoundationPose>

python run_demo.py \
--mesh_file $OBJ_PATH \
--test_scene_dir data/human_demo_raw_data/$TASK_NAME/ \
--debug 0 \
--debug_dir data/human_demo_processed_data/$TASK_NAME/object_pose_trajectory/
```

This creates the following:

```
data/human_demo_processed_data/$TASK_NAME/object_pose_trajectory/
  ├── ob_in_cam
  |   ├── 00000.txt
  |   ├── 00001.txt
  |   └── ...
  └── track_vis
      ├── 00000.png
      ├── 00001.png
      └── ...
```

* `ob_in_cam` contains .txt files representing the 4x4 transformation matrix of the object relative to the camera frame (`T_C_O`).

* `track_vis` contains .png files visualizing the predicted object pose trajectory.

## Predicting Hand Pose Trajectory

The goal of this section is to predict the hand pose trajectory using HaMeR with additional processing to get the hand pose at every timestep. Make sure that you have installed HaMeR depth environment (instructions [here](../../docs/installation.md)). Then run the following:

```
python thirdparty/hamer_depth/run.py \
--rgb-path data/human_demo_raw_data/$TASK_NAME/rgb/ \
--depth-path data/human_demo_raw_data/$TASK_NAME/depth/ \
--mask_path data/human_demo_raw_data/$TASK_NAME/hand_masks/ \
--cam-intrinsics-path data/human_demo_raw_data/$TASK_NAME/cam_K.txt \
--out-path data/human_demo_processed_data/$TASK_NAME/hand_pose_trajectory/
```

This creates the following:

```
data/human_demo_processed_data/$TASK_NAME/hand_pose_trajectory/
  ├── 00000.obj
  ├── 00000.json
  ├── 00000.png
  ├── 00001.obj
  ├── 00001.json
  ├── 00001.png
  └── ...
```

## Retargeting Hand Pose to Robot

Identify the start index of the human demonstration (can tune `--t_offset <time in seconds>` to choose a different start index):

```
python human2sim2robot/human_demo/identify_start_idx.py \
--object-poses-dir data/human_demo_processed_data/$TASK_NAME/object_pose_trajectory/ob_in_cam \
--plot
```

```
export START_IDX=127
```

Visualize the demo data (help to tune the start index):

```
python human2sim2robot/human_demo/visualize_demo.py \
--obj-path $OBJ_PATH \
--object-poses-dir data/human_demo_processed_data/$TASK_NAME/object_pose_trajectory/ob_in_cam \
--hand-poses-dir data/human_demo_processed_data/$TASK_NAME/hand_pose_trajectory/ \
--visualize-table \
--start-idx $START_IDX
```

Retarget hand pose to robot with IK:

```
export ROBOT_FILE=iiwa_allegro.yml

python human2sim2robot/human_demo/retarget_human_to_robot.py \
--obj-path $OBJ_PATH \
--object-poses-dir data/human_demo_processed_data/$TASK_NAME/object_pose_trajectory/ob_in_cam \
--hand-poses-dir data/human_demo_processed_data/$TASK_NAME/hand_pose_trajectory/ \
--visualize-table \
--robot-file $ROBOT_FILE \
--output_filepath data/human_demo_processed_data/$TASK_NAME/retargeted_robot.npz \
--start_idx $START_IDX
```

This creates the following with the retargeted robot trajectory and an interpolated version of this trajectory (and other useful info):

```
data/human_demo_processed_data/$TASK_NAME/retargeted_robot.npz
```

Refer to the main [README](../../README.md) for more visualization instructions.
