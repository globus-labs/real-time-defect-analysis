"""Test drift correction utilities"""
import numpy as np

from pytest import fixture

from rtdefects.drift import compute_drift_from_image_pair, compute_drifts_from_images, subtract_drift_from_images, compute_drifts_from_images_multiref
from rtdefects.analysis import label_instances_from_mask, analyze_defects


@fixture()
def drift_series():
    image = np.zeros((256, 256), dtype=np.int32)
    image[40:44, 60:70] = 1

    # Roll the image progressively in the same direction
    output = [image]
    for i in range(8):
        image = np.roll(image, (6, 10), axis=(0, 1))
        output.append(image)
    return output


def test_drift_from_pair(drift_series):
    drift = compute_drift_from_image_pair(drift_series[0], drift_series[1])
    assert np.isclose([10, 6], drift).all()  # Image coordinates are in width x height

    drift, conv = compute_drift_from_image_pair(drift_series[0], drift_series[1], return_conv=True)
    assert conv.shape == (256, 256)


def test_drift_from_series(drift_series):
    drift = compute_drifts_from_images(drift_series)
    assert np.isclose(np.diff(drift, axis=0), [10, 6]).all()

    # Make sure the coordinate system matches up
    labelled_1 = label_instances_from_mask(drift_series[0], min_size=5)[None, :, :]
    labelled_2 = label_instances_from_mask(drift_series[1], min_size=5)[None, :, :]
    assert labelled_1.mean() > 0
    assert labelled_1.shape == (1, 256, 256)

    details_1, details_2 = map(analyze_defects, [labelled_1, labelled_2])
    assert np.allclose(details_2['positions'][0] - details_1['positions'][0], [10, 6])

    # Apply the corrections and ensure the defects are atop each other
    corrected_images = subtract_drift_from_images(drift_series, drift)
    assert np.allclose(corrected_images, corrected_images[0])


def test_multiref_drift(drift_series):
    drifts = compute_drifts_from_images_multiref(drift_series, lookahead=(1,))
    assert np.isclose(np.diff(drifts, axis=0), [10, 6]).all()
