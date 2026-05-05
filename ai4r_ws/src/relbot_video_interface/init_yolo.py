from ultralytics import YOLO

if __name__ == "__main__":
  model = YOLO("yolov8n.pt")

  model.track(source=0, show=True)