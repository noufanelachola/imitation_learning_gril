import os
import numpy as np


def generate_gril(path: str, file_list: list):

    for file in file_list:

        file_path = os.path.join(path, file)
        images, depths, gazes, actions = read_npz(file_path)

        for i in range(len(images)):
            yield (
                {
                    "image": images[i],
                    "depth": depths[i]
                },
                {
                    "gaze": gazes[i],
                    "action": actions[i]
                }
            )


def read_npz(file_path: str) -> list:
    with np.load(file_path) as data:
        images = data["images"]/255
        depth = data["depth"]
        gazes = data["gaze_coords"]
        actions = data["action"]

        return images, depth, gazes, actions
