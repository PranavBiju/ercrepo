import os
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import rclpy
from rclpy.node import Node

# Load depth model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
SCALE = 1.0  # Change later after calibration

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/path/to/anomaly/images'  # SET THIS
        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        
        # Simulate pose (x, y, z) from SLAM
        self.current_pose = [0.0, 0.0, 1.2]  # Placeholder z: height of camera from ground
        
        self.process_all_images()

    def process_all_images(self):
        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            full_path = os.path.join(self.image_dir, image_file)

            x, y = self.extract_data_from_filename(image_file)
            z = self.calculate_absolute_z(full_path)

            image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            kp, des = self.orb.detectAndCompute(image, None)

            if self.is_duplicate(des, x, y, z):
                self.get_logger().info(f"Duplicate anomaly at ({x:.2f}, {y:.2f}, {z:.2f})")
                continue

            self.declared_anomalies.append({
                'x': x, 'y': y, 'z': z, 'des': des
            })
            self.get_logger().info(f"New anomaly detected at ({x:.2f}, {y:.2f}, {z:.2f})")

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

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()