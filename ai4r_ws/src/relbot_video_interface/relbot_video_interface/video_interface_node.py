#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class VideoInterfaceNode(Node):
    def __init__(self):
        super().__init__('video_interface')
        # Publisher: sends object position to the RELBot
        # Topic /object_position is watched by the robot controller for actuation
        self.position_pub = self.create_publisher(Point, '/object_position', 10)

        # Dataset of detecting people
        # self.model = YOLO("yolov8n.pt")  # default format (slower)
        self.model = YOLO("/ai4r_ws/src/relbot_video_interface/yolov8n_saved_model/yolov8n_float16.tflite")  # tflite format (faster)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

        # Declare GStreamer pipeline as a parameter for flexibility
        self.declare_parameter('gst_pipeline', (
            'v4l2src device=/dev/video0 ! '
            'video/x-raw,width=640,height=480,framerate=30/1 ! '
            'videoconvert ! '
            'video/x-raw,format=RGB ! '
            'appsink name=sink'
        ))
        pipeline_str = self.get_parameter('gst_pipeline').value

        # Initialize GStreamer and build pipeline
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        # Drop late frames to ensure real-time processing
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        # Timer: fires at ~30Hz to pull frames and publish positions
        # The period (1/30) sets how often on_timer() is called
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')
        

    def on_timer(self):
        # Pull the latest frame from the GStreamer appsink
        sample = self.sink.emit('pull-sample')
        if not sample:
            # No new frame available
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            # Failed to map buffer data
            return

        # Convert raw buffer to numpy array [height, width, channels]
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)

        # Display the raw input frame for debugging
        cv2.imshow('Input Stream', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # Run YOLO tracking on the frame and hardhat tracking on the frame
        results = self.model.track(frame, show=True, persist=True, verbose=False)  # Run YOLO tracking on the frame
        #  maybe add? tracker="bytetrack.yaml"

        msg = Point()

        best_id = None
        best_box = None

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()   # shape (N, 4)
                ids = result.boxes.id.cpu().numpy()       # shape (N,)
                classes = result.boxes.cls.cpu().numpy()  # shape (N,)

                for i in range(len(boxes)):
                    cls = int(classes[i])

                    # Only keep persons (class 0 in COCO)
                    if cls != 0:
                        continue

                    track_id = int(ids[i])

                    if best_id is None or track_id < best_id:
                        best_id = track_id
                        best_box = boxes[i]

        if best_box is not None:
            x_center, y_center, w, h = best_box

            #self.get_depth(frame, best_box, show=True)  # uncomment for depth estimation
            # <-- CONVERT DEPTH TO METERS -->

            msg.x = float(x_center)
            msg.y = float(y_center)
            msg.z = float(10000)
        else:
            msg.x = 0.0
            msg.y = 0.0
            msg.z = 0.0

        
        self.position_pub.publish(msg) # Publish the computed position to the robot controller

        self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')


    def get_depth(self, frame, box, show):
        if box is not None:
            return -1
        img_pil = Image.fromarray(frame)

        # Define the transformation for MiDaS model
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # MiDaS model requires input images of size 384x384
            transforms.ToTensor(),
        ])

        # Apply the transformation to the input image
        input_batch = transform(img_pil).unsqueeze(0)

        # Perform depth prediction using MiDaS
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert the depth map tensor to a NumPy array for plotting
        depth_map = prediction.cpu().numpy()

        x_center, y_center, w, h = box
        x1 = x_center - w/2
        x2 = x_center + w/2
        y1 = y_center - h/2
        y2 = y_center + h/2
        object_region_depths = depth_map[int(y1):int(y2), int(x1):int(x2)]
        depth_value = np.mean(object_region_depths)

        if show:
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

            # 2. Convert to 8-bit (0–255)
            depth_8u = (depth_norm * 255).astype(np.uint8)

            # 3. Apply colormap (optional but recommended)
            depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)
            cv2.imshow('Camera', depth_color)
            cv2.waitKey(1)

        return depth_value


    def destroy_node(self):
        # Cleanup GStreamer resources on shutdown
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoInterfaceNode()
    try:
        rclpy.spin(node)  # Keep node alive, invoking on_timer periodically
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()