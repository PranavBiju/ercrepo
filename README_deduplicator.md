
# 🛰️ Anomaly Deduplicator for Panther Bot (ROS 2)

This ROS 2 Python node detects and deduplicates anomalies captured from camera images on the Panther robot. It computes 3D coordinates of anomalies using pre-embedded global (x, y) in the filename and estimated depth (z) using the Depth Anything model. It filters out duplicates using ORB feature descriptors and spatial proximity.

---

## 🚀 Features

- ✅ Global 3D position extraction:
  - `x`, `y` from filename
  - `z` from Depth Anything (MiDaS-based) model
- ✅ ORB-based feature descriptor matching
- ✅ 3D proximity filtering to detect duplicates
- ✅ Fully compatible with ROS 2 (Humble or newer)

---

## 🧱 Folder Structure

```
anomaly_deduplicator/
├── deduplicator_node.py         # The main deduplication node
├── anomaly_images/              # Folder of anomaly image files
└── README.md                    # This file
```

**Image filenames should be in this format:**
```
image_x=<xcoord>_y=<ycoord>.jpg
```

Example:
```
image_x=12.34_y=56.78.jpg
```

---

## 📦 Installation

### 🐍 Python Dependencies

Install Python libraries using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install pillow
pip install transformers
```

> 💡 If using CPU only, remove the `--index-url` flag from the `torch` install command.

### 🧪 ROS 2 (Humble)

Ensure ROS 2 is installed and sourced correctly:

```bash
source /opt/ros/humble/setup.bash
```

You also need to have a ROS 2 workspace. If not already created:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## ⚙️ Setup Instructions

1. **Clone your project to the ROS 2 workspace:**

```bash
cd ~/ros2_ws/src
git clone <your-repo-url> anomaly_deduplicator
```

2. **Edit the `deduplicator_node.py`:**

Set the image folder path:
```python
self.image_dir = '/absolute/path/to/anomaly_images'
```

3. **Build and source workspace:**

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

4. **Run the node:**

```bash
ros2 run anomaly_deduplicator deduplicator_node
```

---

## 📌 Sample Output

```bash
[INFO] [anomaly_deduplicator]: New anomaly detected at (12.34, 56.78, 1.41)
[INFO] [anomaly_deduplicator]: Duplicate anomaly at (12.34, 56.78, 1.41)
```

---

## 🧠 Model Used

- [`LiheYoung/depth-anything-small-hf`](https://huggingface.co/LiheYoung/depth-anything-small-hf)
- Hugging Face `transformers` for depth preprocessing (`AutoImageProcessor`)
- ORB and BFMatcher from `OpenCV`

---

## 📎 Notes

- `SCALE` in the code can be calibrated based on your real-world robot-camera setup.
- `current_pose` contains camera height only, assuming `x` and `y` are from GPS/SLAM.
- This is a pre-processing pipeline; integrate real-time SLAM and camera streams for live deployment.

---

## 📬 Contact

For bugs, suggestions, or questions, raise an issue or email the author.
