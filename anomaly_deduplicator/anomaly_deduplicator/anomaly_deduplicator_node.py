import os
import json
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # SLAM pose message type

# Load depth model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
SCALE = 1.0  # Calibration-dependent

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/path/to/anomaly/images'  # ✅ SET THIS
        self.output_dir = 'deduplicated_anomalies'
        os.makedirs(self.output_dir, exist_ok=True)

        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        self.seen_files = set()

        # Global pose from SLAM
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, z

        # Subscribe to SLAM pose topic
        self.pose_sub = self.create_subscription(
            PoseStamped,           # Change to PoseWithCovarianceStamped if needed
            '/pose',          # ✅ SET THIS to your actual SLAM topic
            self.pose_callback,
            10
        )

        # Check for new images every 2 seconds
        self.timer = self.create_timer(2.0, self.process_new_images)

    def pose_callback(self, msg):
        """Update robot's global pose from SLAM"""
        self.current_pose[0] = msg.pose.position.x
        self.current_pose[1] = msg.pose.position.y
        self.current_pose[2] = msg.pose.position.z  # Often 0 for ground robot

    def process_new_images(self):
        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            if image_file in self.seen_files:
                continue

            self.seen_files.add(image_file)
            full_path = os.path.join(self.image_dir, image_file)

            try:
                # Local coordinates from filename
                rel_x, rel_y = self.extract_data_from_filename(image_file)
                z = self.calculate_absolute_z(full_path)

                # Convert to global using SLAM pose
                x = self.current_pose[0] + rel_x
                y = self.current_pose[1] + rel_y

                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                kp, des = self.orb.detectAndCompute(image, None)

                if self.is_duplicate(des, x, y, z):
                    self.get_logger().info(f"Duplicate anomaly at ({x:.2f}, {y:.2f}, {z:.2f})")
                    continue

                self.declared_anomalies.append({
                    'x': x, 'y': y, 'z': z, 'des': des
                })

                # Save image to output directory
                output_path = os.path.join(self.output_dir, image_file)
                cv2.imwrite(output_path, cv2.imread(full_path))

                # Log it
                self.get_logger().info(f"New anomaly saved: {output_path} at ({x:.2f}, {y:.2f}, {z:.2f})")

                # Save JSON metadata
                self.save_metadata()

            except Exception as e:
                self.get_logger().error(f"Failed to process {image_file}: {e}")

    def extract_data_from_filename(self, filename):
        base = os.path.basename(filename)
        x = float(base.split("x=")[1].split("_")[0])
        y = float(base.split("y=")[1].split("_")[0])
        return x, y

    def calculate_absolute_z(self, image_path):
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
        relative_z = float(np.median(depth_map)) * SCALE
        absolute_z = self.current_pose[2] + relative_z
        return absolute_z

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

    def save_metadata(self):
        metadata = [
            {k: v for k, v in d.items() if k in ['x', 'y', 'z']} for d in self.declared_anomalies
        ]
        with open(os.path.join(self.output_dir, "anomalies_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()