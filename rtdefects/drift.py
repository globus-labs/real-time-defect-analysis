"""Algorithms for correcting drift in microscopy images

Drifts are described using the coordinate system conventions of
`scikit-image <https://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions>`_,
which defines the origin as the top left image is the origin (0, 0).
That means that drifts to the right are in the positive x direction
and drifts downwards are in the positive y direction.
"""
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import fftconvolve
from skimage.transform import AffineTransform, warp


def compute_drifts_from_tracks(tracks: pd.DataFrame, minimum_tracks: int = 1) -> np.ndarray:
    """Estimate the drift for each frame from the positions of voids that were mapped between multiple frames

    We determine the "drift" based on the median displacement of all voids, which is based
    on the assumption that there is no net motion of all the voids.

    We compute the drift in each frame and assume the drift remains unchanged if there are no voids matched
    between a frame and the previous.

    In contrast, trackpy uses the mean and only computes drift when there are matches between frames.

    Args:
        tracks: Track information generated by trackpy.
        minimum_tracks: The minimum number of tracks a void must appear to be used in drift correction
    Returns:
        Drift correction for each frame
    """

    # We'll assume that the first frame has a void
    drifts = [(0, 0)]

    # We're going to go frame-by-frame and guess the drift from the previous frame
    last_frame = tracks.query('frame==0')
    for fid in range(1, tracks['frame'].max() + 1):
        # Join the two frames
        my_frame = tracks.query(f'frame=={fid}')
        aligned = last_frame.merge(my_frame, on='particle')

        # The current frame will be the previous for the next iteration
        last_frame = my_frame

        # If there are no voids in both frames, assign a drift change of 0
        if len(aligned) < minimum_tracks:
            drifts.append(drifts[-1])
            continue

        # Get the median displacements displacements
        last_pos = aligned[['x_x', 'y_x']].values
        cur_pos = aligned[['x_y', 'y_y']].values
        median_disp = np.mean(cur_pos - last_pos, axis=0)

        # Add the drift to that of the previous image
        drift = np.add(drifts[-1], median_disp)
        drifts.append(drift)

    return np.array(drifts)


def compute_drifts_from_images(images: list[np.ndarray], return_conv: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Estimate drift from a stack of images

    Compares adjacent pairs of images in the list

    Args:
        images: Images arranged sequentially
        return_conv: Whether to return the convolution between each pair of images
    """

    # Get the drift between adjacent pairs
    convs = []
    drifts = [(0, 0)]
    for image_1, image_2 in zip(images, images[1:]):
        drift, conv = compute_drift_from_image_pair(image_1, image_2, return_conv=True)
        drifts.append(drift)
        convs.append(conv)

    # Get the cumulative drift
    drifts = np.cumsum(drifts, axis=0)
    if return_conv:
        return drifts, np.array(convs)
    return drifts


def compute_drifts_from_images_multiref(images: list[np.ndarray], offsets: Iterable[int] = (1, 2, 4), pbar: bool = False) -> np.ndarray:
    """Estimate drift for a stack of images by comparing each image to multiple images in the stack

    Estimates a single drift for each image which explains all pairwise comparisons
    made between images in the stack using linear least squares.

    The relative drift between two pairs of images, :math:`\\delta d_{i,j}`, is equal to the
    difference between their absolute drifts, :math:`\\delta d_{i,j} = d_j - d_i`.
    The values of relative drift from observations of :math:`\\delta_{i,j}`
    form a series of linear equations and thus may be solved using linear least squares.

    Args:
        images: Images arranged
        offsets: Compute the drift between each frame and those these number of steps ahead of it in the sequence
        pbar: Whether to display a progress bar

    Returns:
        Drift assumed from all comparisons
    """

    # Compute the number of comparisons
    offsets = list(offsets)
    if any(i <= 0 for i in offsets):
        raise ValueError('All offset values must be positive')
    total_points = len(offsets) * len(images) - sum(offsets)

    # Compute the drift between all pairs
    pair_drifts = np.zeros((total_points, 2))
    a = np.zeros((total_points, len(images)))
    prog_bar = tqdm(total=total_points, disable=not pbar)
    pos = 0
    for step in offsets:
        for i, (image_1, image_2) in enumerate(zip(images, images[step:])):
            pair_drifts[pos, :] = compute_drift_from_image_pair(image_1, image_2)
            a[pos, i], a[pos, i + step] = -1, 1
            prog_bar.update()
            pos += 1
    assert pos == total_points, 'My math for the total number of points was wrong'

    # Solve the least squares problem
    return np.linalg.lstsq(a, pair_drifts, rcond=None)[0]


def compute_drift_from_image_pair(image_1: np.ndarray, image_2: np.ndarray, return_conv: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the drift between two different frames

    Args:
        image_1: Starting image
        image_2: Next image
        return_conv: Whether to return the convolution between the two images
    Returns:
        - The optimal translation between the two images
        - Convolution used to make the judgement, if ``return_conv`` is True
    """

    # Compute the correlations between the two images using FFT
    #  You must reverse the second signal/image for this trick
    conv = fftconvolve(image_1, image_2[::-1, ::-1], mode='same')

    # Find the location of the maximum
    peak_loc = np.unravel_index(np.argmax(conv), conv.shape)

    # Find its displacement from the image center, that's the location
    drift = [peak_loc[1] - conv.shape[0] // 2, peak_loc[0] - conv.shape[1] // 2]
    if return_conv:
        return -np.array(drift), conv
    return -np.array(drift)


def subtract_drift_from_images(images: list[np.ndarray],
                               drifts: np.ndarray,
                               expand_images: bool = False,
                               fill_value: int | float | None = None) -> list[np.ndarray]:
    """Subtract the drift from each image in a series

    Args:
        images: List of images. Assumes the first two dimensions to be the shape
        drifts: Drift computed for the images in the stack
        expand_images: Whether to increase the size of images to accommodate drift
        fill_value: Values to fill in matrix when expanding it
    Returns:
        List of images after correction
    """

    if expand_images:
        # Compute the amount of expansion to make
        min_drift = np.floor(drifts.min(axis=0)).astype(int)
        max_drift = np.ceil(drifts.max(axis=0)).astype(int)
        to_expand = max_drift - min_drift

        expanded_images = []
        for i, (image, drift) in enumerate(zip(images, drifts)):
            # Create the fattened image
            new_size = tuple(np.add(to_expand[::-1], image.shape[:2])) + image.shape[2:]
            new_image = np.zeros(new_size, dtype=image.dtype)
            if fill_value is not None:
                new_image.fill(fill_value)

            # Place the image at the original origin
            shift = max_drift - drift.astype(int)
            new_image[
                shift[1]:shift[1] + image.shape[1],
                shift[0]:shift[0] + image.shape[0]
            ] = image
            expanded_images.append(new_image)
        return expanded_images
    else:
        return [warp(image, AffineTransform(translation=drift), preserve_range=True).astype(image.dtype) for image, drift in zip(images, drifts)]
