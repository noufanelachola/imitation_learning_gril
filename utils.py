import matplotlib.pyplot as plt
import numpy as np

from batch_loader import generate_gril


def visualise(image: np.ndarray, gaze: np.ndarray) -> None:
    """
        shows the image with its gaze point
    """

    gaze = gaze * [image.shape[0], image.shape[1]]

    plt.imshow(image)
    plt.scatter(gaze[0], gaze[1], c='red', s=40, label='Gaze')
    plt.title(f"Gaze at: {gaze}")
    plt.legend()
    plt.axis('off')
    plt.show()


def test_generate_gril(path: str, file_list: list, max: int):
    """
        function to test generate_gril() 
    """

    gen = generate_gril(path, file_list)
    print(gen)

    for i, (x, y) in enumerate(gen):
        print("Sample : {i+1}")
        print(f"Image : {x['image'].shape}")
        print(f"Depth : {x['depth'].shape}")
        print(f"Gaze : {y['gaze']}")
        print(f"Action : {y['action']}")
        print("-"*15)

        if i == max:
            break
