import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from PIL import Image as PILImage
import os
from transformers import AutoProcessor, AutoModelForDepthEstimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("LiheYoung/Depth-Anything-v2-small")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/Depth-Anything-v2-small").to(device).eval()

DUPLICATE_THRESHOLD = 0.5  # meters

# Placeholder camera intrinsics (replace with actual calibrated values)
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5
SCALE = 1.0  # Relative-to-absolute depth scale

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.bridge = CvBridge()
        self.declared_anomalies = []
        self.image_dir = '/path/to/anomaly/images'  # Replace with actual path

        self.get_logger().info("Anomaly Deduplicator Node Started")
        self.process_directory(self.image_dir)

    def extract_data_from_filename(self, filename):
        """Assumes filename contains x=XX_y=YY_u=U_v=V"""
        base = os.path.basename(filename)
        x = float(base.split("x=")[1].split("_")[0])
        y = float(base.split("y=")[1].split("_")[0])
        u = int(base.split("u=")[1].split("_")[0])
        v = int(base.split("v=")[1].split(".")[0])
        return x, y, u, v

    def calculate_absolute_z(self, image_path, u, v):
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

        h, w = depth_map.shape
        if 0 <= v < h and 0 <= u < w:
            d = float(depth_map[v, u])  # relative depth
        else:
            self.get_logger().warn(f"Pixel ({u}, {v}) out of bounds for image: {image_path}")
            d = float(np.median(depth_map))

        z = SCALE * d  # Approximate absolute Z
        return z

    def is_duplicate(self, x, y, z):
        for _, sx, sy, sz in self.declared_anomalies:
            distance = np.sqrt((x - sx)**2 + (y - sy)**2 + (z - sz)**2)
            if distance < DUPLICATE_THRESHOLD:
                return True
        return False

    def process_directory(self, image_dir):
        for file in os.listdir(image_dir):
            if file.endswith((".jpg", ".png")):
                file_path = os.path.join(image_dir, file)
                x, y, u, v = self.extract_data_from_filename(file)
                z = self.calculate_absolute_z(file_path, u, v)

                if not self.is_duplicate(x, y, z):
                    self.declared_anomalies.append((file, x, y, z))
                    self.get_logger().info(f"New anomaly: {file} -> X:{x:.3f}, Y:{y:.3f}, Z:{z:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()