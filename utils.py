import matplotlib.pyplot as plt
import numpy as np


def visualise(image: np.ndarray, coord: np.ndarray) -> None:
    """
        shows the image with its gaze point
    """

    coord = coord * [image.shape[0], image.shape[1]]

    plt.imshow(image)
    plt.imshow(image)
    plt.scatter(coord[0], coord[1], c='red', s=40, label='Gaze')
    plt.title(f"Gaze at: {coord}")
    plt.legend()
    plt.axis('off')
    plt.show()
