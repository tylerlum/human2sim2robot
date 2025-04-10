from pathlib import Path

from setuptools import find_packages, setup

VERSION = "0.1.0"
DESCRIPTION = "Official implementation of Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration"
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="human2sim2robot",
    version=VERSION,
    author="Tyler Lum",
    author_email="tylergwlum@gmail.com",
    url="https://github.com/tylerlum/human2sim2robot",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    keywords=[
        "robotics",
        "dexterous manipulation",
        "simulation",
        "human video demonstration",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    # python_requires=">=3.7, <3.9",  # Comment this out because sometimes need a different Python version, but just can't use isaacgym if use different version
)
