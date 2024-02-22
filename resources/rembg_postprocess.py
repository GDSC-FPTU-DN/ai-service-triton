from typing import List, Any
import numpy as np
import numpy.typing as npt


def rembg_postprocess_processing(args: Any, inputs: List[npt.NDArray]):
    # Get mask
    mask = inputs[0]
    mask = mask[:, 0, :, :]

    # Reverse normalize
    mask_max = np.max(mask)
    mask_min = np.min(mask)
    mask = (mask - mask_min) / (mask_max - mask_min)

    # Reverse scale
    mask = np.squeeze(mask) * 255
    mask = mask.astype("uint8")

    return (mask,)
