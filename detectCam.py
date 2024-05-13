import sys
sys.path.append('../')
import os
import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.patches as patches
import torch
from utils.detect import FaceDetector
from model.mtcnn_pytorch import PNet, RNet,ONet
import matplotlib.pyplot as plt
import time
import argparse


# Define command-line arguments
parser = argparse.ArgumentParser(description='Face Detection with MTCNN')
parser.add_argument('-stage', dest="use3stage",default=2, type=int, help="use3stage = 3 for 3 Stage MTCNN, use3stage = 2 for 2Stage MTCNN")
parser.add_argument('-jit', dest="use_jit",default=True, type=bool, help="if True, use JIT")
parser.add_argument('-quantized', dest="use_quantized_rnet",default=True, type= bool, help="Use Quantized RNET model if True")
args = parser.parse_args()

if args.use_quantized_rnet:
    mname = "rnet_quantized.pth"
else:
    mname = "rnet_sc.pth"

if args.use3stage == 3:
    use3stage = True
elif args.use3stage == 2:
    use3stage = False
else:
    raise("Select Either 2 or 3 for -stage argument")
# Load the pre-trained models
pnet = torch.jit.load("output/trainedModels/pnet_sc_v2.pth")
rnet = torch.jit.load(f"output/trainedModels/{mname}")
onet = torch.jit.load("output/trainedModels/onet_sc.pth")

# Initialize the face detector
fd_jit = FaceDetector(pnet=pnet, rnet=rnet, onet=onet, device="cpu", prof=False, use3stage=use3stage, use_jit=args.use_jit)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Variables for FPS calculation
start_time = time.time()
frame_count = 0
i = 0
total_fps = 0
# Main loop to capture frames from the webcam
while True:
    i+=1
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Convert the frame to PIL image
    pil_frame = Image.fromarray(frame)

    # Perform face detection
    boxes_sc = fd_jit.detect(frame)

    # Draw rectangles around the detected faces
    draw = ImageDraw.Draw(pil_frame)
    for box in boxes_sc:
        x1, y1, x2, y2 = box[:4]  # Extracting the coordinates from the box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    # Convert the PIL image back to OpenCV format
    frame = np.array(pil_frame)

    # Calculate FPS every 2 seconds
    frame_count += 1
    #if time.time() - start_time > 1:
    fps = frame_count / (time.time() - start_time)
    total_fps+=fps
    start_time = time.time()
    frame_count = 0

    # Display FPS on top right corner of the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
print(f"Average FPS: {total_fps/i}")
cap.release()
cv2.destroyAllWindows()
