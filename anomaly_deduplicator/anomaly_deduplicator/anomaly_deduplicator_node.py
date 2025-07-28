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
import tf_transformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
SCALE = 1.0

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/home/smartnihar/ros2_ws/src/anomaly_frames'
        self.output_dir = '/home/smartnihar/ros2_ws/src/anomaly_unique'
        os.makedirs(self.output_dir, exist_ok=True)

        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        self.seen_files = set()

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

    def image_quality(self, gray, keypoints):
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 0.7 * sharpness + 0.3 * len(keypoints)

    def find_duplicate_index(self, new_des, x, y, z, pos_threshold=3000, match_threshold=30):
        for i, anomaly in enumerate(self.declared_anomalies):
            dx = anomaly['x'] - x
            dy = anomaly['y'] - y
            dz = anomaly['z'] - z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < pos_threshold:
                if anomaly.get('des') is not None and new_des is not None:
                    matches = self.bf.match(anomaly['des'], new_des)
                    if len(matches) >= match_threshold:
                        return True, i
        return False, -1

    def update_anomaly(self, idx, x, y, z, des, quality, area, idno, image_file, full_path):
        self.declared_anomalies[idx].update({
            'x': x,
            'y': y,
            'z': z,
            'des': des,
            'quality': quality,
            'area': area,
            'idno': idno,
            'filename': image_file
        })
        dst_path = os.path.join(self.output_dir, image_file)
        cv2.imwrite(dst_path, cv2.imread(full_path))
        self.get_logger().info(
            f"Replaced representative for anomaly #{idx} with higher-quality, larger-area image: {image_file}"
        )
        self.save_metadata()

    def process_new_images(self):
        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            full_path = os.path.join(self.image_dir, image_file)

            try:
                idno, area, rel_x, rel_y = self.extract_data_from_filename(image_file)
                z = self.calculate_absolute_z(full_path)

                cos_yaw = np.cos(self.current_yaw)
                sin_yaw = np.sin(self.current_yaw)
                global_dx = rel_x * cos_yaw - rel_y * sin_yaw
                global_dy = rel_x * sin_yaw + rel_y * cos_yaw

                x = self.current_pose[0] + global_dx
                y = self.current_pose[1] + global_dy

                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                kp, des = self.orb.detectAndCompute(image, None)
                quality = self.image_quality(image, kp)

                is_dup, idx = self.find_duplicate_index(des, x, y, z)

                if is_dup:
                    prev_area = self.declared_anomalies[idx].get('area', 0)
                    if area > prev_area:
                        self.delete_nearby_idnos(idno)
                        self.update_anomaly(idx, x, y, z, des, quality, area, idno, image_file, full_path)
                    else:
                        self.get_logger().info(
                            f"Duplicate (lower quality/area) discarded: {image_file} at ({x:.2f}, {y:.2f}, {z:.2f})"
                        )
                    continue

                self.declared_anomalies.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'des': des,
                    'quality': quality,
                    'area': area,
                    'idno': idno,
                    'filename': image_file
                })
                cv2.imwrite(os.path.join(self.output_dir, image_file), cv2.imread(full_path))
                self.get_logger().info(f"New anomaly saved: {image_file} at ({x:.2f}, {y:.2f}, {z:.2f})")
                self.save_metadata()

            except Exception as e:
                self.get_logger().error(f"Error processing {image_file}: {e}")

    def delete_nearby_idnos(self, idno, delta=10):
        files = os.listdir(self.output_dir)
        for fname in files:
            if not fname.endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                fid = int(fname.split("_")[0])
                if abs(fid - idno) <= delta:
                    os.remove(os.path.join(self.output_dir, fname))
                    self.get_logger().info(f"Deleted old nearby image due to better anomaly: {fname}")
            except Exception as e:
                self.get_logger().warning(f"Could not parse ID from filename {fname}: {e}")

    def extract_data_from_filename(self, filename):
        base = os.path.basename(filename)
        idno = int(base.split("_")[0])
        area = float(base.split("a=")[1].split("_")[0])
        x = float(base.split("x=")[1].split("_")[0])
        y = float(base.split("y=")[1].split("_")[0])
        return idno, area, x, y

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

    def is_duplicate(self, new_des, x, y, z, pos_threshold=3000, match_threshold=30):
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
            {k: v for k, v in d.items() if k in ['x', 'y', 'z', 'area', 'idno']}
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