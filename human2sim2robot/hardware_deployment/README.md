# Hardware Deployment

The goal of this section is to deploy the policy in the real world. For this step, you need to have a policy in `data/sim_training/checkpoints/`.

You likely will need 2 or 3 machines.

1. Robot Computer: This is the computer that directly controls the robot.

2. Perception Computer: This is the computer that runs the perception nodes.

3. Policy Computer: This is the computer that runs the policy nodes.

Make sure these are all on the same network. Then for each machine, run the following:

```
export ROS_MASTER_URI=http://<perception_computer_hostname>:11311  # e.g., export ROS_MASTER_URI=http://bohg-ws-X.stanford.edu:11311
export ROS_HOSTNAME=$(hostname)

# OPTIONAL: export HOME=<path_to_home>  # Where ROS logs and stuff will be stored, useful if this is a different location from default
```

First, we need to set important ROS parameters.

On the perception computer, run the following:

```
rosparam set /mesh_file <path_to_mesh_file>  # e.g., rosparam set /mesh_file /home/tyler/github_repos/human2sim2robot/assets/kiri/snackbox/snackbox.obj
rosparam set /text_prompt <text_prompt>  # e.g., rosparam set /text_prompt "red crackerbox"
```

If using zed camera:

```
rosparam set /camera_type zed
roslaunch zed_wrapper zed.launch
```

If using realsense camera:

```
rosparam set /camera_type realsense
roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud
```

On the perception computer, run the following:

```
python thirdparty/segment-anything-2-real-time/sam2_ros_node.py
python thirdparty/FoundationPose/fp_ros_node.py
python thirdparty/FoundationPose/fp_evaluator_ros_node.py
rqt_image_view & rqt_image_view & rqt_image_view &
```

On the robot computer, run the following (make sure you have set up the iiwa (https://github.com/tylerlum/iiwa_hardware) and allegro hand (https://github.com/HaozhiQi/ros-allegro)):

```
roslaunch iiwa_control joint_position_control.launch
roslaunch allegro_hand allegro_hand.launch HAND:=right AUTO_CAN:=true CONTROLLER:=pd
python human2sim2robot/hardware_deployment/move_arm_ros_node.py  # Modify this file to move to the pre-manipulation arm pose

# You can also manually jog the robot arm to the pre-manipulation arm pose
# You can also manually move the robot hand to the pre-manipulation hand pose with the following:
rostopic pub /allegroHand_0/joint_cmd sensor_msgs/JointState "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
position: [-0.24632971, 0.3363568, 0.34074978, 0.17196047, 0.11733355, 0.47271622, 0.19272655, 0.06724613, 0.46699552, 0.4426419, 0.29637433, 0.13175883, 1.21938345, 0.08001256, 0.87144459, 0.02593612]  # Replace this with the pre-manipulation hand pose
velocity: []
effort: []"
```

NOTE: If you want to test this ROS pipeline, but run it using simulation instead of real robot, you can instead run:

```
python human2sim2robot/hardware_deployment/sim_ros_node.py  # Modify this file to load the correct sim parameters
```

On the policy computer, run the following:
```
python human2sim2robot/hardware_deployment/visualizer_ros_node.py --load_point_cloud True
python human2sim2robot/hardware_deployment/rl_policy_ros_node.py  # Modify this file to load the correct policy checkpoint and config
python human2sim2robot/hardware_deployment/fabric_ros_node.py
python human2sim2robot/hardware_deployment/fabric_upsampler_ros_node.py
python human2sim2robot/hardware_deployment/goal_pose_ros_node.py  # Modify this file to load the correct goal poses
```

Flags to vary what is visualized in visualizer.
