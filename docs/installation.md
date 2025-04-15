# Installation

We recommend using `mamba` to install the dependencies for faster installation, but you can replace `mamba` with `conda` if you prefer.

Install `mamba`:
```
# Miniconda install https://docs.anaconda.com/miniconda/#quick-command-line-install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

# Mamba install https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
conda install mamba -c conda-forge
```

Note: Because of some dependency conflicts, we need to use different environments for different parts of the framework.

# Fabrics

We use a similar Geometric Fabric controller to the one used in [DextrAH-G](https://sites.google.com/view/dextrah-g). As of April 2025, this code is not currently open-source. It should be open-sourced soon! It is still possible to use this codebase without this, but you need to disable the Geometric Fabric and replace some of the Geometric Fabric code used for forward kinematics.

# Primary Environment

Primary environment (for human-to-robot retargeting, visualization, simulation training, etc.):
```
mamba create -n human2sim2robot_env python=3.8
mamba activate human2sim2robot_env

# Torch + cuda
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

## Ffmpeg for visualization
conda install -c conda-forge ffmpeg

# Misc dependencies
pip install matplotlib numpy scipy pybullet tqdm tyro pynput transforms3d cached_property_with_invalidation wandb tensorboard tensorboardX ruff omegaconf hydra-core urdfpy gym trimesh imageio'[ffmpeg]' live_plotter isaacgym-stubs open3d

# Clone the repository
git clone https://github.com/tylerlum/human2sim2robot.git
cd human2sim2robot
export ROOT_DIR=$(pwd)
export THIRD_PARTY_DIR=$ROOT_DIR/thirdparty

mkdir $THIRD_PARTY_DIR

# Curobo custom fork (Library Installation step in https://curobo.org/get_started/1_install_instructions.html#library-installation)
cd $THIRD_PARTY_DIR
git clone https://github.com/tylerlum/curobo.git
cd curobo
git checkout Human2Sim2Robot # Last tested on commit hash 00e8c612790f0b2b5cddb78bc17f861203b46e6d, may still work on more updated versions
conda install -c conda-forge git-lfs; git lfs pull  # Maybe need to add this (https://github.com/NVlabs/curobo/issues/10)
pip install -e . --no-build-isolation  # ~20 min

# Fabrics
cd $THIRD_PARTY_DIR
git clone https://gitlab.com/tylerlum/fabrics-sim.git
cd fabrics-sim
git checkout Human2Sim2Robot # Last tested on commit hash 9b542217d1e615cc1892faa92238b6647716d507, may still work on more updated versions
pip install -e .
chmod +x urdfpy_patch.sh
./urdfpy_patch.sh  # This is only needed for Python 3.10+, but it doesn't hurt to run it

# Isaacgym (https://developer.nvidia.com/isaac-gym)
# Must extract the file "IsaacGym_Preview_4_Package.tar.gz" to $THIRD_PARTY_DIR/isaacgym
cd $THIRD_PARTY_DIR/isaacgym/python
pip install -e .
pip install numpy==1.23.5  # Compatible with isaacgym (np.float removed in 1.24)

# Human2Sim2Robot
cd $ROOT_DIR
pip install -e .
```

Notes:

* isaacgym requires Python 3.7 or 3.8

* The installation has been tested on Ubuntu 20.04 with a NVIDIA RTX 4090 GPU. The installation should work on similar setups, but has not been tested extensively.

* You may need to run the following if you have issacgym issues (e.g., when running on a cluster or cloud compute):

```
# Isaacgym things
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/isaacgym_modified/isaacgym/python/isaacgym/_bindings/linux-x86_64

sudo apt-get install gcc-11 g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

sudo apt-get update \
 && sudo apt-get install vulkan-tools \
 && sudo apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs
```

* Common issue with curobo installation:

```
[5/5] c++ geom_cuda.o sphere_obb_kernel.cuda.o pose_distance_kernel.cuda.o self_collision_kernel.cuda.o -shared -L/home/tylerlum/miniconda3/envs/human2sim2robot_env/lib/python3.8/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/home/tylerlum/miniconda3/envs/human2sim2robot_env/lib64 -lcudart -o geom_cu.so
FAILED: geom_cu.so 
c++ geom_cuda.o sphere_obb_kernel.cuda.o pose_distance_kernel.cuda.o self_collision_kernel.cuda.o -shared -L/home/tylerlum/miniconda3/envs/human2sim2robot_env/lib/python3.8/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/home/tylerlum/miniconda3/envs/human2sim2robot_env/lib64 -lcudart -o geom_cu.so
/usr/bin/ld: cannot find -lcudart
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.
```

Just try to pip install curobo again.

```
cd $THIRD_PARTY_DIR/curobo
pip install -e . --no-build-isolation
```

* Another common issue with curobo installation:

```
    132 | #error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

```

Fixed with:

```
sudo apt-get install gcc-11 g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

OR

```
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=x86_64-conda-linux-gnu-g++
```

* If you encounter a segfault like `segmentation fault (core dumped)`, it could be from a few different things:

  * It could be from trying to run the sim training with a viewer (`headless=False`) from a system without a monitor (e.g., a cluster node)
  * It could be from some curobo issue. You can retry building curobo `pip install -e . --no-build-isolation` or turn off curobo `USE_CUROBO=False`
  * It could be from issues with collisions (objects spawning in other objects, meshes too complex, resulting in too many collisions). You can try to avoid spawning in collision or increasing sim parameters like `default_buffer_size_multiplier` and `max_gpu_contact_pairs`.
 
* If you encounter an error like `malloc(): invalid size (unsorted)`, it could be from:

  * `gym.begin_aggregate` and `gym.end_aggregate` not working right, can just not use these functions

# Real-World Evaluation Environment

Create the environment for real-world evaluation, which requires ROS Noetic.

## Option 1: System-wide ROS Installation

Just use the same environment as the primary environment, but install ROS following the documentation [here](https://wiki.ros.org/noetic/Installation/Ubuntu).

This is the only option that can use integrate ROS with isaacgym, which allows sim-in-the-loop evaluation (ROS nodes for control, but replace real robot and environment with isaacgym simulation
).

```
sudo apt install ros-noetic-rqt-image-view
sudo apt install ros-noetic-rqt-plot
```

## Option 2: RoboStack ROS Installation

Sometimes system-wide ROS installation fails or may cause conflicts with other parts of your machine. [RoboStack](https://robostack.github.io/index.html) allows you to install ROS in a conda/mamba environment. This is great, but one frustrating issue is that it only works with certain Python versions (often the newest ones), so it is not compatible with the primary environment (requires Python 3.8).

We will first start by following the RoboStack ROS Noetic installation instructions [here](https://robostack.github.io/GettingStarted.html), which is copied below.

```
mamba create -n human2sim2robot_ros_env python=3.11
mamba activate human2sim2robot_ros_env

# RoboStack
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
mamba install ros-noetic-desktop

mamba deactivate
mamba activate human2sim2robot_ros_env
```

Then install the following:

```
# (COPY FROM PRIMARY ENV) Torch + cuda
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# (COPY FROM PRIMARY ENV) Curobo custom fork (Library Installation step in https://curobo.org/get_started/1_install_instructions.html#library-installation)
cd $THIRD_PARTY_DIR
git clone https://github.com/tylerlum/curobo.git
cd curobo
git checkout Human2Sim2Robot # Last tested on commit hash c8edf2cc9a10d639700271196c00ffda1e8c358f, may still work on more updated versions
git lfs pull  # Maybe need to add this (https://github.com/NVlabs/curobo/issues/10)
pip install -e . --no-build-isolation  # ~20 min

# (COPY FROM PRIMARY ENV) Fabrics
cd $THIRD_PARTY_DIR
git clone https://gitlab.com/tylerlum/fabrics-sim.git
cd fabrics-sim
git checkout Human2Sim2Robot # Last tested on commit hash 9b542217d1e615cc1892faa92238b6647716d507, may still work on more updated versions
pip install -e .
chmod +x urdfpy_patch.sh
./urdfpy_patch.sh  # This is only needed for Python 3.10+, but it doesn't hurt to run it

# (COPY FROM PRIMARY ENV) Human2Sim2Robot
cd $ROOT_DIR
pip install -e .

# Misc
pip install git+https://github.com/tylerlum/fast-simplification.git

# May need to do this to make sure this didn't get broken by fast-simplification
pip install numpy==1.23.5
pip install pybullet --reinstall

# ROS tools
mamba install ros-noetic-rqt-image-view
mamba install ros-noetic-rqt-plot
```

# Other Environments

We need to use different environments the following:

* Object and Hand Segmentation using Segment Anything Model 2 (SAM2):

```
cd $THIRD_PARTY_DIR
git clone https://github.com/tylerlum/segment-anything-2-real-time.git
cd segment-anything-2-real-time
git checkout Human2Sim2Robot  # Last tested on commit hash d4ea81a1b9f3a43fb996443368cfce64c3c58c4e
```

* Object Pose Estimation using FoundationPose:

```
cd $THIRD_PARTY_DIR
git clone https://github.com/tylerlum/FoundationPose.git
cd FoundationPose
git checkout Human2Sim2Robot  # Last tested on commit hash ee6ef78555298a07b55f9e0f9bc0f12be546b663
```

* Hand Pose Estimation using HaMeR:

```
cd $THIRD_PARTY_DIR
git clone https://github.com/tylerlum/human_shadow.git
cd human_shadow
git checkout Human2Sim2Robot  # Last tested on commit hash e2fb81e3a339185d1f0764da3167607b31608d92
```

Their individual installation instructions are on their respective GitHub repositories.
