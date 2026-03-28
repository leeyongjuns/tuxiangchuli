Image Matching and Stitching Experiment

Overview

This experiment focuses on feature detection, feature matching, and image stitching.

The goal is to understand how different feature extraction methods perform and how they can be used to align and stitch images.

🛠️ Tech Stack
Python
OpenCV
NumPy

Implementations
1. Harris Corner Detection
Detects corner features in images
Produces many feature points but may include incorrect matches
2. SIFT (Scale-Invariant Feature Transform)
Detects stable and distinctive keypoints
Provides high matching accuracy
3. Feature Matching
Matches keypoints between two images
Used to find corresponding regions
4. Image Stitching
Aligns and merges two images into one
Based on matched feature points

Tasks
Capture two real images (not cropped from one image)
Perform feature detection and matching
Compare Harris and SIFT performance
Generate stitched image

Analysis
Harris:
Produces many keypoints
High mismatch rate
SIFT:
Fewer keypoints
Much higher accuracy

SIFT performs better in complex structures due to its robustness and stability.
