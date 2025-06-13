import cv2 as cv
import numpy as np


def reshape_depth(depth):
    width = 224
    height = 224
    frame = depth
    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
    frame = np.expand_dims(frame, axis=2)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0


def reshape_image(image):
    width = 224
    height = 224
    frame = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0


def arilNN(airsim_img, depth, aril):
    img = reshape_image(np.float32(airsim_img))
    depth = reshape_depth(np.float32(depth))
    input_data = [img, depth]

    commands, gaze = aril.predict(input_data)
    print("Commands", commands)
    return commands, gaze
