# Stereo Visual Odometry

Stereo Visual Odometry is a Python project for estimating the 3D motion trajectory of a stereo camera rig using visual odometry techniques.
![Data GIF](https://drive.google.com/file/d/1Bx3anHUfq9Da-cmyGPJAbshQdU1nXT7E/view?usp=sharing)

## Overview

Visual odometry is a key component in robotics and computer vision for estimating the motion of a camera in 3D space based on visual input from a sequence of images. Stereo Visual Odometry specifically leverages stereo camera setups to estimate the camera's pose and reconstruct the 3D environment.

This project implements stereo visual odometry using feature matching, triangulation, and pose estimation techniques. It processes stereo image pairs to estimate the camera's trajectory and visualize the path taken.

## Features

- **Feature Matching**: Detects and matches keypoints between stereo image pairs.
- **Triangulation**: Computes 3D point cloud reconstruction using stereo correspondences.
- **Pose Estimation**: Estimates the camera's motion trajectory using visual odometry techniques.
- **Visualization**: Visualizes the estimated camera path in a 3D plot.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Usage

To use Stereo Visual Odometry, follow these steps:

1. Clone the repository to your local machine.
   ```
   git clone https://github.com/ekrrems/Stereo-Visual-Odometry
   ```
3. Install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```
4. Prepare stereo image pairs for processing.
5. Run the main script to perform stereo visual odometry.
    ```bash
    python main.py
    ```
6. Optionally, modify parameters and settings in the script to customize the odometry process.

## Licence
This project is licenced under the MIT Licence
