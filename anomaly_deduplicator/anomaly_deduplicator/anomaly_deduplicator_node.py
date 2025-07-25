import os
import json
import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations  # for quaternion to euler conversion

# Load depth model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
SCALE = 1.0  # adjust based on calibration

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/path/to/anomaly/images'  # 🔁 Set your folder path
        self.output_dir = 'deduplicated_anomalies'
        os.makedirs(self.output_dir, exist_ok=True)

        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        self.seen_files = set()

        # Robot's global pose and orientation
        self.current_pose = [0.0, 0.0, 0.0]
        self.current_yaw = 0.0

        self.pose_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(2.0, self.process_new_images)

    def odom_callback(self, msg: Odometry):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        self.current_pose[2] = msg.pose.pose.position.z

        orientation_q = msg.pose.pose.orientation
        quat = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        self.current_yaw = yaw

    def process_new_images(self):
        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            if image_file in self.seen_files:
                continue

            self.seen_files.add(image_file)
            full_path = os.path.join(self.image_dir, image_file)

            try:
                rel_x, rel_y = self.extract_data_from_filename(image_file)
                z = self.calculate_absolute_z(full_path)

                # ✅ Apply rotation from robot's yaw to local offset
                cos_yaw = np.cos(self.current_yaw)
                sin_yaw = np.sin(self.current_yaw)
                global_dx = rel_x * cos_yaw - rel_y * sin_yaw
                global_dy = rel_x * sin_yaw + rel_y * cos_yaw

                # ✅ Global position
                x = self.current_pose[0] + global_dx
                y = self.current_pose[1] + global_dy

                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                kp, des = self.orb.detectAndCompute(image, None)

                if self.is_duplicate(des, x, y, z):
                    self.get_logger().info(f"Duplicate anomaly at ({x:.2f}, {y:.2f}, {z:.2f})")
                    continue

                self.declared_anomalies.append({'x': x, 'y': y, 'z': z, 'des': des})
                cv2.imwrite(os.path.join(self.output_dir, image_file), cv2.imread(full_path))
                self.get_logger().info(f"New anomaly saved: {image_file} at ({x:.2f}, {y:.2f}, {z:.2f})")
                self.save_metadata()

            except Exception as e:
                self.get_logger().error(f"Error processing {image_file}: {e}")

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
            {k: v for k, v in d.items() if k in ['x', 'y', 'z']}
            for d in self.declared_anomalies
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