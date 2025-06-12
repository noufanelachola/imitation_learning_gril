import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def gril():
    mobilenet = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None
    )

    mobilenet.trainable = False

    rgb = Input(shape=(224, 224, 3), name="image")

    x = mobilenet(rgb, training=False)
    x = Conv2D(64, (5, 5), strides=2, padding="same", activation="relu")(x)
    x = Conv2D(64, (5, 5), strides=2, padding="same", activation="relu")(x)
    pool11 = MaxPool2D(pool_size=(2, 2))(x)
    rgb_flat = Flatten()(pool11)

    depth = Input(shape=(224, 224, 1), name="depth")

    d = Conv2D(64, (5, 5), strides=2, padding='same', activation='relu')(depth)
    d = Conv2D(64, (5, 5), strides=2, padding='same', activation='relu')(d)
    d = Conv2D(32, (5, 5), strides=2, padding='same', activation='relu')(d)
    d = Conv2D(16, (5, 5), strides=2, padding='same', activation='relu')(d)
    pool21 = MaxPool2D(pool_size=(2, 2))(d)
    depth_flat = Flatten()(pool21)

    shared = Concatenate()([rgb_flat, depth_flat])

    a = Dense(512, activation='elu')(shared)
    a = Dense(256, activation='elu')(a)
    a = Dense(128, activation='elu')(a)
    a = Dense(64, activation='elu')(a)
    action = Dense(4, name="action")(a)

    g = Dense(512, activation='relu')(shared)
    g = Dense(256, activation='relu')(g)
    g = Dense(128, activation='relu')(g)
    g = Dense(64, activation='relu')(g)
    gaze = Dense(2, name='gaze')(g)

    model = Model(inputs=[rgb, depth], outputs=[action, gaze])
    model.summary()

    return model
