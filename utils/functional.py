import cv2
import torch
import numpy as np
import sys
from numba import jit
import ctypes


def cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


@jit
def cpu_nms_jit(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

@jit
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr


def nms(dets, scores, thresh, use_jit = True):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    if use_jit:

        dets = np.concatenate([
            dets.astype(np.float32),
            scores.astype(np.float32).reshape(-1, 1)
        ], 1)


        ret = cpu_nms_jit(dets,thresh)

    else:
        dets = np.concatenate([
            dets.astype(np.float32),
            scores.astype(np.float32).reshape(-1, 1)
        ], 1)
        ret = cpu_nms(dets, thresh)

    return ret


def imnormalize(img):
    """
    Normalize pixel value from (0, 255) to (-1, 1) 
    """

    img = (img - 127.5) * 0.0078125
    return img




# Define the Box structure
class Box(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float),
                ("y1", ctypes.c_float),
                ("x2", ctypes.c_float),
                ("y2", ctypes.c_float)]

# Load the shared library
lib = ctypes.CDLL("utils/nms.so")

# Define the argument and return types for the nms function
lib.nms.argtypes = [ctypes.POINTER(Box), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
lib.nms.restype = ctypes.POINTER(ctypes.c_int)

def nms_c(dets, scores, iou_threshold):
    # Convert bounding boxes to Box structure
    boxes = (Box * len(dets))()
    for i, (x1, y1, x2, y2) in enumerate(dets):
        boxes[i].x1 = x1
        boxes[i].y1 = y1
        boxes[i].x2 = x2
        boxes[i].y2 = y2
    
    # Convert scores to ctypes array
    scores = np.array(scores, dtype=np.float32)
    scores_ptr = ctypes.cast(scores.ctypes.data, ctypes.POINTER(ctypes.c_float))
    
    # Call the nms function
    selected_count = ctypes.c_int()
    selected_indices_ptr = lib.nms(boxes, scores_ptr, len(dets), iou_threshold, ctypes.byref(selected_count))

    # Convert selected indices to a Python list
    selected_indices = [selected_indices_ptr[i] for i in range(selected_count.value)]

    # Free allocated memory
    libc = ctypes.CDLL("libc.so.6")
    libc.free(selected_indices_ptr)

    # Return selected detections
    return [dets[i] for i in selected_indices]