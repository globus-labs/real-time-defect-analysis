"""Test drift correction utilities"""
import numpy as np

from rtdefects.drift import compute_drift_from_images


def test_drift_from_images():
    image = np.zeros((256, 256))
    image[40:44, 60:70] = 1

    shifted_image = np.roll(image, (6, 10), axis=(0, 1))

    drift = compute_drift_from_images(image, shifted_image)
    assert np.isclose([10, 6], drift).all()  # Image coordinates are in width x height
