Image Interpolation Experiment

Overview

This experiment focuses on image interpolation techniques for recovering missing or damaged pixels.

The goal is to understand how different interpolation methods work and compare their performance.

🛠️ Tech Stack
Python
OpenCV
NumPy
Matplotlib

Implementations
1. Nearest Neighbor Interpolation
Assigns pixel value from the nearest known pixel
Fast but produces blocky results
2. Bilinear Interpolation
Uses 4 neighboring pixels to compute new pixel values
Produces smoother results than nearest neighbor
3. RBF Interpolation
Uses radial basis functions for interpolation
Produces smooth and continuous results but computationally expensive

Tasks
Task 1: Image Damage and Recovery
Randomly draw lines on images to simulate damage
Recover missing pixels using:
Nearest Neighbor
Bilinear
RBF interpolation
Task 2: Pixel Loss Experiment
Randomly remove 10% ~ 90% of pixels
Recover images using interpolation methods
Evaluate using:
SSIM
L2 Norm

Analysis
Nearest Neighbor: Fast but low-quality reconstruction
Bilinear: Balanced between quality and speed
RBF: Best quality but highest computational cost

Interpolation quality decreases as pixel loss increases.
