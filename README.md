# Edge Detection Experiment using OpenCV

## Overview

This project implements multiple edge detection algorithms using OpenCV.

The goal of this experiment is to understand and compare different edge detection methods, including:

* Prewitt
* Sobel
* Canny
* FDoG (Flow-based Difference of Gaussian)

Through this experiment, we focus on understanding how each algorithm works and their differences.

---

## Tech Stack

* Python
* OpenCV
* NumPy

---

## Implementations

### 1. Prewitt

* Implemented using custom convolution kernels
* Detects edges based on horizontal and vertical gradients

### 2. Sobel

* Uses OpenCV built-in function
* More robust to noise compared to Prewitt

### 3. Canny

* Multi-stage edge detection algorithm
* Includes noise reduction, gradient calculation, and thresholding

### 4. FDoG

* Advanced edge detection method based on flow fields
* Produces more coherent and stylized edges

---

## Analysis

* **Prewitt**: Simple but sensitive to noise
* **Sobel**: More stable due to smoothing effect
* **Canny**: Produces thin and accurate edges
* **FDoG**: Generates smooth edges but edge thickness may vary

FDoG produces edges with varying width due to its flow-based filtering process.


