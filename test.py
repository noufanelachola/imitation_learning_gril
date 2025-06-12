import numpy as np
from utils import visualise

data = np.load("training_data/flipped_truck_mountains10.npz")
keys = data.files

print("Keys : ", keys)

for key in keys:
    print(key, data[key].shape)

for i in range(2):
    print(data["images"][i].shape)
    visualise(data["images"][i], data["gaze_coords"][i])
