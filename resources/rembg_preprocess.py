from typing import Any, List
import numpy as np
import numpy.typing as npt


def rembg_preprocess_initialize(args: Any):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return {
        "mean": mean,
        "std": std
    }


def rembg_preprocess_processing(args: Any, inputs: List[npt.NDArray]):
    # Input image
    img = inputs[0]

    # Scale Img
    img = img / np.max(img)

    # Normalize Img with formula
    img[:, :, :, 0] = (img[:, :, :, 0] - args['mean'][0]) / args['std'][0]
    img[:, :, :, 1] = (img[:, :, :, 1] - args['mean'][1]) / args['std'][1]
    img[:, :, :, 2] = (img[:, :, :, 2] - args['mean'][2]) / args['std'][2]

    # Transpose Img to (channel, width, height)
    img = img.transpose((0, 3, 1, 2))

    # Expand dims for batch_size axis
    return (img.astype(np.float32),)
