#!/usr/bin/env python3
import numpy as np


def tile_stacked_images(stacked_images):
    # 9 x 84 x 84 -> 84 x 84 x 9 -> 84 x (84x3) x 3
    # chw --> hwc
    assert isinstance(stacked_images, np.ndarray)
    assert stacked_images.shape[0] % 3 == 0, f'Expects the first dim to be multiple of 3 but got {stacked_images.shape[0]}'
    n_channels = stacked_images.shape[0] // 3
    stacked_images = stacked_images.transpose(1, 2, 0)  # chw --> hwc
    imgs = [stacked_images[:, :, 3*i:3*(i+1)] for i in range(n_channels)]
    return np.concatenate(imgs, axis=1)
