import os
import numpy as np
import cv2
import torch
import shutil
import json
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  # SLAM pose

# Load depth model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
SCALE = 1.0  # Adjust based on calibration

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/path/to/anomaly/images'  # <-- CHANGE THIS
        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        self.latest_pose = None  # Updated via SLAM subscriber

        self.create_subscription(Odometry, '/slam/odom', self.slam_callback, 10)

        # Wait for SLAM pose before processing
        self.timer = self.create_timer(1.0, self.check_pose_ready)

    def slam_callback(self, msg: Odometry):
        self.latest_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z
        }

    def check_pose_ready(self):
        if self.latest_pose:
            self.timer.cancel()
            self.get_logger().info("SLAM pose received, starting processing...")
            self.process_all_images()

    def process_all_images(self):
        save_dir = os.path.join(self.image_dir, "deduplicated_anomalies")
        os.makedirs(save_dir, exist_ok=True)
        metadata = []

        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            full_path = os.path.join(self.image_dir, image_file)

            x = self.latest_pose['x']
            y = self.latest_pose['y']
            z = self.calculate_absolute_z(full_path, self.latest_pose['z'])

            image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            kp, des = self.orb.detectAndCompute(image, None)

            if self.is_duplicate(des, x, y, z):
                self.get_logger().info(f"Duplicate anomaly at ({x:.2f}, {y:.2f}, {z:.2f})")
                continue

            self.declared_anomalies.append({
                'x': x, 'y': y, 'z': z, 'des': des
            })

            # Save deduplicated image
            save_path = os.path.join(save_dir, image_file)
            shutil.copy(full_path, save_path)

            metadata.append({'file': image_file, 'x': x, 'y': y, 'z': z})
            self.get_logger().info(f"New anomaly at ({x:.2f}, {y:.2f}, {z:.2f})")

        # Save metadata
        with open(os.path.join(save_dir, "anomalies_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def calculate_absolute_z(self, image_path, robot_z):
        image = PILImage.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        rel_z = float(np.median(depth_map)) * SCALE
        abs_z = robot_z + rel_z  # Global Z = SLAM Z + relative camera depth
        return abs_z

    def is_duplicate(self, new_des, x, y, z, pos_threshold=0.3, match_threshold=30):
        for anomaly in self.declared_anomalies:
            dx = anomaly['x'] - x
            dy = anomaly['y'] - y
            dz = anomaly['z'] - z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            if dist < pos_threshold:
                if anomaly['des'] is not None and new_des is not None:
                    matches = self.bf.match(anomaly['des'], new_des)
                    if len(matches) >= match_threshold:
                        return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()