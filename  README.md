
# Anomaly Deduplication Tool

This repository contains a Python script to detect and remove duplicate anomaly entries based on 3D proximity. It uses the MiDaS depth estimation model and assumes anomaly images are saved with global coordinates in their filenames (e.g., `anomaly_x=3.21_y=7.12.png`).

## Features

- Estimates depth (Z coordinate) using MiDaS model
- Calculates 3D distance between anomalies
- Avoids reporting near-duplicate anomalies
- Can be extended with ROS 2, but also works as standalone Python script

---

## Setup Instructions

### 🛠️ Dependencies

Make sure you have the following installed:

- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/locally/)
- OpenCV
- `cv_bridge` (optional, used only if integrating with ROS2)
- ROS 2 (optional)
- Other Python packages

You can install all required packages using pip:

```bash
pip install torch torchvision opencv-python pillow numpy
```

> 🔧 Note: `cv_bridge` is only needed if you’re running this in ROS2. If you're using ROS2, install it via your ROS distro (e.g., `sudo apt install ros-humble-cv-bridge`).

---

## 🔍 MiDaS Model Setup

This script uses the [MiDaS](https://github.com/isl-org/MiDaS) depth estimation model provided by Intel ISL via `torch.hub`.

You do not need to download the model manually. It will be automatically fetched on first run using:

```python
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
```

Make sure you have internet access the first time you run the script.

---

## 📁 Directory Structure

```
anomaly-deduplicator/
├── deduplicator.py        # Main Python script
├── README.md              # This file
└── /images                # Folder containing anomaly images
```

---

## 📸 Image Naming Convention

Anomaly image files must be named like this:

```
anomaly_x=3.21_y=7.12.png
```

The `x` and `y` coordinates will be extracted from the filename. The `z` coordinate (depth) will be calculated using MiDaS.

---

## 🚀 Running the Script (Standalone)

Update the image directory path in `deduplicator.py`:

```python
self.image_dir = '/full/path/to/anomaly/images'
```

Then run:

```bash
python deduplicator.py
```

---

## 🤖 Running with ROS 2 (Optional)

If you’re integrating with ROS 2, this script is already set up as a node:

```bash
ros2 run <your_package> deduplicator
```

Make sure your ROS 2 workspace is built and sourced properly.

---



