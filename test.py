import numpy as np
import os

from utils import visualise
from utils import test_generate_gril

path = "training_data/train"
file_list = os.listdir(path)

# for i, file in enumerate(file_list):
#     data = np.load(os.path.join(path, file))
#     print(f"File {i+1} : {data['images'].shape[0]}")


data = np.load("training_data/train/flipped_truck_mountains10.npz")
keys = data.files

print("Keys : ", keys)

for key in keys:
    print(key, data[key].shape)

for i in range(2):
    print(data["images"][i].shape)
    visualise(data["images"][i], data["gaze_coords"][i])

# test_generate_gril(path, file_list, 3)
