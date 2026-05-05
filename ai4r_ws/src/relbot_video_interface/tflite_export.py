from ultralytics import YOLO

if __name__ == "__main__":
  model = YOLO("yolov8n.pt")

  model.export(format="tflite", imgsz=(320,240))