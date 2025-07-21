
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
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()

# Load transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

DUPLICATE_THRESHOLD = 0.5  # meters

class AnomalyDeduplicator(Node):

    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.bridge = CvBridge()
        self.declared_anomalies = []
        self.image_dir = '/path/to/anomaly/images'

        self.get_logger().info("Anomaly Deduplicator Node Started")
        self.process_directory(self.image_dir)

    def extract_coordinates(self, filename):
        base = os.path.basename(filename)
        x = float(base.split("x=")[1].split("_")[0])
        y = float(base.split("y=")[1].split(".")[0])
        return x, y

    def calculate_depth(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = PILImage.fromarray(img)
        input_tensor = transform(input_img).to(midas.device)

        with torch.no_grad():
            prediction = midas(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return float(np.median(depth_map))

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
                x, y = self.extract_coordinates(file)
                z = self.calculate_depth(file_path)

                if not self.is_duplicate(x, y, z):
                    self.declared_anomalies.append((file, x, y, z))
                    self.get_logger().info(f"New anomaly: {file} -> X:{x}, Y:{y}, Z:{z}")

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
