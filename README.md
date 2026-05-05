## Right after opening the container
`pip3 install lap --break-system-packages`

Then run (from ai4r_ws)
`python3 src/relbot_video_interface/init_yolo.py`

After that this line should work fine
`ros2 launch relbot_video_interface video_interface.launch.py`

For testing purposes the Gstreamer pipeline to receive RELBot camera feed was replaced with webcam capture. Go to launch/video_interface.launch.py and uncomment and comment out parameters part to use the normal pipeline.

I exported the model to tflite so that it runs faster on CPU. As far as I know, there are no drowbacks to doing this, so I would recommend using it. If you want to use GPU, you might then want to use the default model format. For that, uncomment this line and comment out the line after it.
`self.model = YOLO("yolov8n.pt")`

## for exporting (should be unnecessary if using provided exported models)
`pip3 install "tensorflow>=2.0.0,<=2.19.0" --break-system-packages`
`pip3 install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71" "onnxruntime" --break-system-packages`
`pip3 install --no-cache-dir "tf_keras<=2.19.0" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0" "onnx2tf>=1.26.3,<1.29.0" --extra-index-url https://pypi.ngc.nvidia.com --break-system-packages`

And run
`python3 src/relbot_video_interface/tflite_export.py`

## To run with depth perception
`apt remove python3-tqdm`
`pip3 install timm --break-system-packages`

Run
`python3 src/relbot_video_interface/init_midas.py`
Wait a bit and press enter? Then type y as many times as requested

Uncomment `self.get_depth(frame, best_box, show=True)`
and run
`ros2 launch relbot_video_interface video_interface.launch.py`
Dont forget to implement conversion of depth to meters