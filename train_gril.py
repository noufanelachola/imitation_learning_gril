import tensorflow as tf
import numpy as np
import random
import os

from batch_loader import generate_gril
from utils import total_samples_count

batch_size = 32
train_datapath = "training_data/train"
val_datapath = "training_data/val"

file_list = os.listdir(train_datapath)
val_list = os.listdir(val_datapath)

random.shuffle(file_list)

total_train_samples = total_samples_count(train_datapath, file_list)
total_val_samples = total_samples_count(val_datapath, val_list)

train_steps_per_epoch = int(np.ceil(total_train_samples/batch_size))
val_steps_per_epoch = int(np.ceil(total_val_samples/batch_size))

print("Converting Train dataset")
tfx = tf.data.Dataset.from_generator(
    generate_gril,
    args=[train_datapath, file_list],
    output_signature=(
        {
            "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            "depth": tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        }, {
            "gaze": tf.TensorSpec(shape=(2,), dtype=tf.float32),
            "action": tf.TensorSpec(shape=(4,), dtype=tf.float32)
        }
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Converting Val dataset")
val = tf.data.Dataset.from_generator(
    generate_gril,
    args=[val_datapath, val_list],
    output_signature=(
        {
            "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            "depth": tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        }, {
            "gaze": tf.TensorSpec(shape=(2,), dtype=tf.float32),
            "action": tf.TensorSpec(shape=(4,), dtype=tf.float32)
        }
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)
