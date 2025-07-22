import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from torchvision.transforms import Compose
from PIL import Image as PILImage
import os

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Load transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

DUPLICATE_THRESHOLD = 0.5  # meters

class AnomalyDeduplicator(Node):

    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.bridge = CvBridge()
        self.declared_anomalies = []
        self.image_dir = '/path/to/anomaly/images'  # 🔁 Replace with actual path

        self.get_logger().info("Anomaly Deduplicator Node Started")
        self.process_directory(self.image_dir)

    def extract_pixel_coordinates(self, filename):
        """Extract pixel (u, v) from filename of format: ...x=U_y=V.png"""
        base = os.path.basename(filename)
        u = int(base.split("x=")[1].split("_")[0])
        v = int(base.split("y=")[1].split(".")[0])
        return u, v

    def calculate_depth(self, image_path, u, v):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = PILImage.fromarray(img_rgb)
        input_tensor = transform(input_img).to(device)

        with torch.no_grad():
            prediction = midas(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        h, w = depth_map.shape
        if 0 <= v < h and 0 <= u < w:
            z = float(depth_map[v, u])
        else:
            self.get_logger().warn(f"Pixel ({u}, {v}) out of bounds for image: {image_path}")
            z = float(np.median(depth_map))  # Fallback

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
                u, v = self.extract_pixel_coordinates(file)
                z = self.calculate_depth(file_path, u, v)

                # Use pixel coords as dummy world coords for now
                x = u
                y = v

                if not self.is_duplicate(x, y, z):
                    self.declared_anomalies.append((file, x, y, z))
                    self.get_logger().info(f"New anomaly: {file} -> X:{x}, Y:{y}, Z:{z:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()