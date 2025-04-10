# Real To Sim

The goal of this section is to create a digital twin of the robot, objects, camera, and environment.

## Robot

First, find or create a URDF file for your robot. Next, we need to configure the robot for use with cuRobo. Follow these instructions: https://curobo.org/tutorials/1_robot_configuration.html#tut-robot-configuration. You can ignore the steps about isaac sim and usd, as we do not use those. The most time consuming part is modeling the robot as a set of collision spheres. This is a manual process, but you can use the following script to help visualize the collision spheres to make this easier (the isaac sim method did not work for me, so I used this instead):

For example, to visualize the collision spheres for the default robot, run:
```
python human2sim2robot/real_to_sim/visualize_robot.py \
--robot_file iiwa_allegro.yml \
--visualize_collision_spheres
```

## Object

We need to create a digital twin of the object.

If you have an existing 3D model, you can use that directly. For example, you can use the IKEA models by navigating to the website (e.g., https://www.ikea.com/us/en/p/kalas-plate-mixed-colors-80461380/), viewing page source (CTRL + U), and searching for ".glb" to find a link to the 3D model. This can be downloaded and then converted to .obj and .mtl files using tools like: https://imagetostl.com/convert/file/glb/to/obj

If you do not have an existing 3D model, we can create one by using [PolyCam](https://poly.cam/) or [Kiri Engine](https://www.kiriengine.app/) to capture the object with the lidar scan mode using an iPhone or iPad with the LiDAR sensor. 

After this, you should have a .obj, .mtl, and .png file. Save the .obj, .mtl, and .png files to `assets/my_object_raw/` like so:

```
assets/my_object_raw/
  ├── my_object.obj
  ├── my_object.mtl
  └── my_object.png
```

Note: You cannot simply rename the files. You need to modify the .obj file to properly point to the associated .mtl file and modify the .mtl file to properly point to the associated .png file. If needed, you can open the file with Blender to manually modify the mesh.

At this point, you should visualize the object to make sure it looks like what you expect. We recommend the following viewers for obj files:

* [C3D](https://www.creators3d.com/online-viewer): Pass in the .obj file. Then you can press "grid" to see the origin frame and dimensions to see the size of the object, which is useful for understanding the object origin and size.

* [3D Viewer](https://www.3dviewer.net/): Pass in multiple files (all of the .obj files, .mtl files, and .png files for the object). This gives a good visualization of the object.

![image](https://github.com/user-attachments/assets/c1b9ad44-61e7-4540-9016-6b8d3200bd7a)

We want our objects to be z-up and have the origin frame at the center of the object. Often, the object will be given as y-up, so we need to rotate it to z-up. We also want to create a .urdf file that can be used in simulation.

Run the following if the object is currently y-up:

```
python human2sim2robot/real_to_sim/process_obj.py \
--obj_path assets/my_object_raw/my_object.obj \
--output_dir assets/my_object \
--create_urdf \
--center_origin \
--current_up_dir y \
--new_up_dir z
```

Run the following if the object is already z-up:
```
python human2sim2robot/real_to_sim/process_obj.py \
--obj_path assets/my_object_raw/my_object.obj \
--output_dir assets/my_object \
--create_urdf \
--center_origin
```

This will create:

```
assets/my_object/
  ├── my_object.urdf
  ├── my_object.obj
  ├── my_object.mtl
  └── my_object.png
```

Lastly, confirm that the object is z-up and centered by visualizing it with the online viewer above.

![image](https://github.com/user-attachments/assets/fed6dad1-f9ef-4881-a84e-00074801f6aa)

![image](https://github.com/user-attachments/assets/ca42b607-9bda-4a51-aa1f-84195a4d2b45)

## Camera

We need to acquire the extrinsic and intrinsic parameters of the camera. The intrinsic parameters are typically given by the camera manufacturer, and the extrinsic parameters are typically given by the camera calibration process.

**Zed Camera**:

* Follow instructions here to set up ZED camera with ROS: https://github.com/stereolabs/zed-ros-wrapper

* `roslaunch zed_wrapper zed.launch`

* `rostopic list` to see the topics, should see something like `/zed/zed_node/rgb/camera_info` for the intrinsic parameters and `/zed/zed_node/rgb/image_raw` for the rgb images and `/zed/zed_node/point_cloud/cloud_registered` for the point cloud

* `rostopic echo /zed/zed_node/rgb/camera_info` this will show you the intrinsic parameters

**Realsense Camera**:

* Follow instructions here to set up Realsense camera with ROS: https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy OR use robostack `mamba install ros-noetic-realsense2-camera`

* `roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud`

* `rostopic list` to see the topics, should see something like `/camera/color/camera_info` for the intrinsic parameters and `/camera/color/image_raw` for the rgb images and `/camera/depth/color/points` for the point cloud

* `rostopic echo /camera/color/camera_info` this will show you the intrinsic parameters

Note: For some reason, the point cloud is not in the camera frame, but a rotated version of it. See code below for how to resolve it.

```
# ZED_CAMERA_T_C_Cptcloud
# For zed, point cloud frame is camera frame with X forward and Y left
# https://community.stereolabs.com/t/coordinate-system-of-pointcloud/908/2
ZED_CAMERA_T_C_Cptcloud = np.eye(4)
ZED_CAMERA_T_C_Cptcloud[:3, :3] = np.array(
    [
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ]
)
ZED_CAMERA_T_R_Cptcloud = ZED_CAMERA_T_R_C @ ZED_CAMERA_T_C_Cptcloud

# REALSENSE_CAMERA_T_C_Cptcloud
# For realsense, it seems that point cloud frame is same as camera frame
REALSENSE_CAMERA_T_R_Cptcloud = REALSENSE_CAMERA_T_R_C
```

Lastly, we need to perform extrinsic calibration, which finds the 4x4 transform between the camera frame and the robot base frame. This is done by putting an apriltag on the robot end effector such that it is visible from the camera. Then we move the robot around to different joint angles, record the joint angles, and save the image of the AprilTag when it is at that position. This gives us a set of positions of the end effector in camera frame and in robot frame that we can use to perform extrinsic calibration. We assume that the robot base frame is z-up and that the camera frame is z-forward, y-down. Reference: https://github.com/droid-dataset/droid/blob/main/docs/example-workflows/calibrating-cameras.md

Note on transforms: We use the notation `T_A_B` to mean the transform of frame B with respect to frame A. The extrinsics calibration gives us `T_R_C`, which is the transform of the camera frame with respect to the robot base frame. We can use this to transform points from the camera frame to the robot base frame.

## Environment

Install the [3D Scanner App](https://3dscannerapp.com/). Use the lidar scan to capture the full scene of the robot and environment. Next, use the edit crop tool to crop out irrelevant parts of the scene, save the result, and rename the file to "my_scene_with_robot". Next, make a copy of the file, crop out the robot, and rename the file to "my_scene_without_robot". Save both files to .obj files, rename them to "my_scene_with_robot.obj" and "my_scene_without_robot.obj", and put them in the `assets/` folders so it looks like (make sure to modify the .obj files to properly point to the associated .mtl file and modify the .mtl file to properly point to the associated .png file):

```
assets/my_scene/
  ├── my_scene_with_robot.obj
  ├── my_scene_with_robot.mtl
  ├── my_scene_with_robot.png
  ├── my_scene_without_robot.obj
  ├── my_scene_without_robot.mtl
  └── my_scene_without_robot.png
```

Next, we need to compute the transform between the robot base frame and the frame of the .obj file. We do this by using ICP registration to find where the robot base mesh is with respect to the .obj file with the robot in it. Since the scene with and without the robot are aligned, we can use the result to properly place the scene with respect to the robot in simulation.

To see an example of this, run:

```
python human2sim2robot/real_to_sim/icp_registration/run.py
```

![image](https://github.com/user-attachments/assets/8d870eff-3af6-4b53-822f-709a24bcf426)

![image](https://github.com/user-attachments/assets/6e96d4ba-b0ff-46c5-87c9-e89b5f7b21c8)

![image](https://github.com/user-attachments/assets/44b253f3-0914-47b6-9b9a-97235e7c7a37)

![image](https://github.com/user-attachments/assets/5da83389-dc54-46f5-b1f2-0135946c2b30)


Press the x button to close the viewer to go to the next step to see the progression.

Now, you can run this for yourself (tune the bounding box and init rotations to get a good result):
```
python human2sim2robot/real_to_sim/icp_registration/run.py \
--source_path human2sim2robot/real_to_sim/icp_registration/robot_base.stl \
--target_path assets/my_scene/my_scene_with_robot.obj \
--output_path assets/my_scene/transform.txt \
--bounding_box_x -0.3 \
--bounding_box_y -0.2 \
--bounding_box_z 0.33 \
--bounding_box_len_x 0.5 \
--bounding_box_len_y 0.5 \
--bounding_box_len_z 0.3 \
--init_roll_deg -90.0 \
--init_pitch_deg 0.0 \
--init_yaw_deg 0.0
```

Manually adjust the bounding box to crop out the robot base from the scene and the init rotations make sure the robot base mesh orientation matches the robot base from the scene. It will show you the input to ICP registration and the output transformation and result. Iteratively run this until the result looks good, and then save out the 4 x 4 transformation matrix to `assets/my_scene/transform.txt`.

Lastly, run the following:

```
python human2sim2robot/real_to_sim/process_scene.py \
--scene_dir assets/my_scene/ \
--transform_path assets/my_scene/transform.txt \
--output_dir assets/my_scene_processed \
--inverse_transform \
--create_urdf
```

This will create:

```
assets/my_scene_processed/
  ├── my_scene_with_robot.urdf
  ├── my_scene_with_robot.obj
  ├── my_scene_with_robot.mtl
  ├── my_scene_with_robot.png
  ├── my_scene_without_robot.urdf
  ├── my_scene_without_robot.obj
  ├── my_scene_without_robot.mtl
  └── my_scene_without_robot.png
```

This transforms the .obj files so that their origin frame aligns with the robot base frame, and we create .urdf files that can be used in simulation.

If you don't want to do this, you can also simply use a tape-measure to manually measure the components of your scene and create your own .obj and .urdf files (e.g., table, wall, box, etc.).

