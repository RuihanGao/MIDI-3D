from typing import List, Tuple

import numpy as np
import PIL
import torch.nn.functional as F
from PIL import Image


def generate_dense_grid_points(
    bbox_min: np.ndarray, bbox_max: np.ndarray, octree_depth: int, indexing: str = "ij"
):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def crop_and_pad(
    image: PIL.Image.Image,
    mask: PIL.Image.Image,
    padding_ratio: float = 0.1,
    padding_value: float = 1.0,
) -> Tuple[PIL.Image.Image]:
    image = (np.array(image) / 255.0).astype(np.float32)
    mask = np.array(mask).astype(np.float32)

    # crop
    coords = np.argwhere(mask > 0.5)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    rgb_cropped = image[y_min : y_max + 1, x_min : x_max + 1]
    mask_cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

    h, w = rgb_cropped.shape[:2]

    # padding
    padding_size = [0, 0, 0, 0]  # top, bottom, left, right
    if w > h:
        padding_size[0] = padding_size[1] = int((w - h) / 2)
        h = w
    else:
        padding_size[2] = padding_size[3] = int((h - w) / 2)
        w = h

    padding_size = tuple([s + int(w * padding_ratio) for s in padding_size])
    rgb_padded = np.pad(
        rgb_cropped,
        (
            (padding_size[0], padding_size[1]),
            (padding_size[2], padding_size[3]),
            (0, 0),
        ),
        mode="constant",
        constant_values=padding_value,
    )
    mask_padded = np.pad(
        mask_cropped,
        (
            (padding_size[0], padding_size[1]),
            (padding_size[2], padding_size[3]),
        ),
        mode="constant",
        constant_values=0,
    )

    # convert to PIL image
    rgb_padded = Image.fromarray((rgb_padded * 255).astype(np.uint8))
    mask_padded = Image.fromarray((mask_padded * 255).astype(np.uint8))

    return rgb_padded, mask_padded
