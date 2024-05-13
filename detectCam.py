import sys
sys.path.append('../')
import os
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import torch
from utils.detect import FaceDetector
from model.mtcnn_pytorch import PNet, RNet,ONet
import matplotlib.pyplot as plt
import time


pnet = torch.jit.load("output/trainedModels/pnet_sc_v2.pth")
rnet = torch.jit.load("output/trainedModels/rnet_sc_v2.pth")
onet = torch.jit.load("output/trainedModels/onet_sc.pth")

# Initialize the face detector
fd_jit = FaceDetector(pnet=pnet, rnet=rnet_q, onet=onet, device="cpu", prof=True, use3stage=False, use_jit=True)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# Main loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame from webcam.")
        break

    # Detect faces in the frame
    boxes_sc = fd_jit.detect(frame)

    # Draw rectangles around the detected faces
    for box in boxes_sc:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()