# Sim Training

The goal of this section is to train a policy to perform a task in simulation.

## Sim Keyboard Commands

* "R" to reset the environment
* "E" to toggle debug visualizations
* "V" to toggle viewer sync
* "B" to breakpoint
* Arrow keys + Page Up/Down to apply forces to the object

In `human2sim2robot/sim_training/tasks/cross_embodiment/env.py`, see `KeyboardShortcut` for all keyboard shortcuts.

## Parameter Details

Some useful things you can tune:
```
task.randomize=True \
randomization_params=RandomizationParams_medium \
task.sim.enable_viewer_sync_at_start=False \
seed=42 \
task.env.custom.OBSERVED_OBJECT_UNCORR_POS_NOISE=0.02 \
task.env.custom.OBSERVED_OBJECT_UNCORR_RPY_DEG_NOISE=15.0 \
task.env.custom.OBSERVED_OBJECT_CORR_POS_NOISE=0.02 \
task.env.custom.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE=15.0 \
task.env.custom.OBSERVED_OBJECT_RANDOM_POSE_INJECTION_PROB=0.1 \
task.env.custom.reset_object_sample_noise_x=0.1 \
task.env.custom.reset_object_sample_noise_y=0.1 \
task.env.custom.reset_object_sample_noise_z=0.01 \
task.env.custom.reset_object_sample_noise_roll_deg=0.0 \
task.env.custom.reset_object_sample_noise_pitch_deg=0.0 \
task.env.custom.reset_object_sample_noise_yaw_deg=20.0 \
task.env.custom.reset_right_robot_sample_noise_arm_deg=2.0 \
task.env.custom.reset_right_robot_sample_noise_hand_deg=2.0 \
task.env.custom.randomForces.forceProb=0.05 \
task.env.custom.randomForces.forceScale=50 \
```

Next, we describe some useful/important parameters:

* `test`: If true, will not train, but will test the policy.
* `num_envs`: Number of environments to run in parallel.
* `train.player.deterministic`: If true, will use a deterministic policy (always use the mean action), else will use a stochastic policy (sample actions from the policy distribution).
* `headless`: If true, will not show the viewer.
* `task.sim.enable_viewer_sync_at_start`: If true, will sync the viewer to the simulation (press V to toggle).
* `task.env.custom.enableDebugViz`: If true, will enable debug visualization (press E to toggle).
* `checkpoint`: Path to the checkpoint to load (either a local path or a wandb URL).
* `train`: The training algorithm to use (e.g. `CrossEmbodimentPPOLSTM`, `CrossEmbodimentPPO`).
* `task`: The task to use (e.g. `CrossEmbodiment`).
* `task.randomize`: If true, will turn on domain randomization.
* `randomization_params`: The randomization parameters to use (e.g. `RandomizationParams_empty`, `RandomizationParams_tiny`, `RandomizationParams_small`, `RandomizationParams_medium`, `RandomizationParams_large`, `RandomizationParams_huge`).
* `task.env.custom.object_urdf_path`: The path to the object URDF file.
* `task.env.custom.retargeted_robot_file`: The path to the retargeted robot file.
* `task.env.custom.object_poses_dir`: The path to the directory containing the object poses.
* `experiment`: The name of the experiment (e.g. `Experiment_snackbox_push`).
* `task.env.custom.USE_FABRIC_ACTION_SPACE`: If true, will use the fabric action space. Else use PD target action space.
* `task.env.custom.ENABLE_FABRIC_COLLISION_AVOIDANCE`: If true, will enable fabric collision avoidance.
* `task.env.custom.FABRIC_CSPACE_DAMPING`: The damping value for the fabric collision avoidance.
* `task.env.custom.FABRIC_CSPACE_DAMPING_HAND`: The damping value for the fabric collision avoidance for the hands.
* `task.env.controlFrequencyInv`: The inverse of the control frequency (e.g. 4 means simulation takes 4 steps between each policy control step).
* `task.env.custom.USE_CUROBO`: If true, will use curobo to smartly randomize the robot position wrt the object.
* `seed`: The random seed to use.
* `task.env.custom.OBSERVED_OBJECT_UNCORR_POS_NOISE`: The noise to add to the observed object position (uncorrelated over time, sampled at every timestep).
* `task.env.custom.OBSERVED_OBJECT_UNCORR_RPY_DEG_NOISE`: The noise to add to the observed object roll, pitch, yaw (uncorrelated over time, sampled at every timestep).`
* `task.env.custom.OBSERVED_OBJECT_CORR_POS_NOISE`: The noise to add to the observed object position (correlated over time, sampled once at the start of the episode).
* `task.env.custom.OBSERVED_OBJECT_CORR_RPY_DEG_NOISE`: The noise to add to the observed object roll, pitch, yaw (correlated over time, sampled once at the start of the episode).
* `task.env.custom.OBSERVED_OBJECT_RANDOM_POSE_INJECTION_PROB`: The probability of injecting random poses into the object pose observation.
* `task.env.custom.reset_object_sample_noise_x`: The noise to add to the object position x (reset distribution).
* `task.env.custom.reset_object_sample_noise_y`: The noise to add to the object position y (reset distribution).
* `task.env.custom.reset_object_sample_noise_z`: The noise to add to the object position z (reset distribution).
* `task.env.custom.reset_object_sample_noise_roll_deg`: The noise to add to the object roll (reset distribution).
* `task.env.custom.reset_object_sample_noise_pitch_deg`: The noise to add to the object pitch (reset distribution).
* `task.env.custom.reset_object_sample_noise_yaw_deg`: The noise to add to the object yaw (reset distribution).
* `task.env.custom.reset_right_robot_sample_noise_arm_deg`: The noise to add to the right robot arm (reset distribution).
* `task.env.custom.reset_right_robot_sample_noise_hand_deg`: The noise to add to the right robot hand (reset distribution).
* `task.env.custom.randomForces.forceProb`: The probability of applying random forces onto the object.
* `task.env.custom.randomForces.forceScale`: The scale of the random forces.
* `device_id`: The GPU device to use (e.g. 0).
* `graphics_device_id`: The GPU device to use for rendering (e.g. 0).
* `asymmetric_critic`: The asymmetric critic method to use (e.g. `AsymmetricCritic_empty`, `AsymmetricCritic_mlp`, `AsymmetricCritic_lstm`).
* `task.env.custom.object_friction`: The friction of the object.
* `task.env.custom.object_mass_scale`: The mass scale of the object.
* `task.env.custom.object_inertia_scale`: The inertia scale of the object.
* `task.env.custom.right_robot_friction`: The friction of the right robot.
* `task.env.custom.table_friction`: The friction of the table.
* `wandb_activate`: If true, will activate wandb.
* `wandb_group`: The name of the wandb group
* `wandb_name`: The name of the wandb run
* `wandb_entity`: The name of the wandb entity (e.g., `tylerlum`)
* `wandb_project`: The name of the wandb project (e.g., `cross_embodiment`)

If you want to do single-node multi-GPU training with `export NUM_GPUS=4` GPUs, replace `python` with `torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_addr 127.0.0.1` and include `multi_gpu=True` in the commandline args.

## Config

The config management can look intimidating at first, but it's not too bad.

`human2sim2robot/sim_training/cfg/config.yaml` is the main config file, which points to other config files.

`human2sim2robot/sim_training/cfg/task/CrossEmbodiment.yaml` is the task config file.

`human2sim2robot/sim_training/cfg/train/CrossEmbodimentPPOLSTM.yaml` is the training config file.

These yaml files are nice, but they can be hard to develop with because there's no syntax highlighting or autocomplete or typehinting in the code. Thus, we also have `human2sim2robot/sim_training/tasks/cross_embodiment/config.py`, which is a python file that uses dataclasses to model the (mostly) same structure as the config files (with no actual data), which also gives us nice syntax highlighting and autocomplete and typehinting in the code.

Also, we have code to check that these files are consistent (e.g., if you add a new parameter in `CrossEmbodiment.yaml` called `my_param`, you need to add it to `config.py` as well, and if you misspell it as `my_paramm` in `CrossEmbodiment.yaml`, it will give you an error).
