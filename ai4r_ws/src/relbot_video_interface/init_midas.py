import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
  midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
  cam = cv2.VideoCapture(0)


  while True:
    ret, frame = cam.read()

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
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert the depth map tensor to a NumPy array for plotting
    depth_map = prediction.cpu().numpy()

    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

    # 2. Convert to 8-bit (0–255)
    depth_8u = (depth_norm * 255).astype(np.uint8)

    # 3. Apply colormap (optional but recommended)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    # Display the captured frame
    cv2.imshow('Camera', depth_color)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
      break

  # Release the capture and writer objects
  cam.release()
  cv2.destroyAllWindows()