import tensorflow as tf
import numpy as np


def action_loss(y_true, y_pred):
    squared_difference = tf.square(y_true-y_pred)
    weights = np.array([0.10, 0.10, 0.10, 0.70])
    weighted_square_difference = weights*squared_difference

    return tf.reduce_mean(weighted_square_difference, axis=-1)
