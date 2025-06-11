# import tensorflow as tf
import numpy as np
import os

from batch_loader import generate_gril

path = "training_data"
file_list = os.listdir(path)

gen = generate_gril(path, file_list)

for i, (x, y) in enumerate(gen):
    print(f"Sample {i+1}")
    print("Image shape:", x["image"].shape)
    print("Depth shape:", x["depth"].shape)
    print("Action:", y["action"])
    print("Gaze:", y["gaze"])
    print("-" * 40)

    if i == 2:
        break
