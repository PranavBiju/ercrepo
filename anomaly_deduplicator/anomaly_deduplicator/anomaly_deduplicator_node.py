import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from PIL import Image as PILImage
import os
from transformers import AutoProcessor, AutoModelForDepthEstimation

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("LiheYoung/Depth-Anything-v2-small")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/Depth-Anything-v2-small").to(device).eval()

# Parameters
DUPLICATE_DISTANCE_THRESHOLD = 0.5  # meters
DUPLICATE_VISUAL_THRESHOLD = 0.8    # cosine similarity score
SCALE = 1.0  # camera-dependent scaling factor (tune this)

class HybridAnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('hybrid_anomaly_deduplicator')
        self.bridge = CvBridge()
        self.image_dir = '/path/to/anomaly/images'  # Set this
        self.declared_anomalies = []  # [(filename, x, y, z, descriptor)]
        self.orb = cv2.ORB_create(500)
        self.current_pose = (0.0, 0.0, 0.0)

        # SLAM Odometry subscriber
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info("Hybrid Deduplicator Node Started")
        self.process_directory(self.image_dir)

    def odom_callback(self, msg):
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )

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

    def compute_visual_descriptor(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        kp, des = self.orb.detectAndCompute(img, None)
        return des

    def is_duplicate(self, x, y, z, descriptor):
        for _, sx, sy, sz, sdesc in self.declared_anomalies:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
            if dist < DUPLICATE_DISTANCE_THRESHOLD:
                if sdesc is not None and descriptor is not None:
                    sim = self.feature_similarity(descriptor, sdesc)
                    if sim > DUPLICATE_VISUAL_THRESHOLD:
                        return True
        return False

    def feature_similarity(self, des1, des2):
        if des1 is None or des2 is None:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if not matches:
            return 0.0
        distances = [m.distance for m in matches]
        return 1.0 - (np.mean(distances) / 256.0)

    def process_directory(self, image_dir):
        for file in os.listdir(image_dir):
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(image_dir, file)
                x, y = self.extract_data_from_filename(file)
                z = self.calculate_absolute_z(path)
                desc = self.compute_visual_descriptor(path)

                if not self.is_duplicate(x, y, z, desc):
                    self.declared_anomalies.append((file, x, y, z, desc))
                    self.get_logger().info(f"New anomaly: {file} at X:{x:.2f} Y:{y:.2f} Z:{z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = HybridAnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()