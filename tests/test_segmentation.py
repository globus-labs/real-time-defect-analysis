"""Test the segmentation function"""
from pathlib import Path

from pytest import fixture, mark
from imageio import v3 as iio
import numpy as np

from rtdefects.analysis import analyze_defects
from rtdefects.segmentation.pytorch import PyTorchSegmenter
from rtdefects.segmentation import download_model
from rtdefects.segmentation.tf import TFSegmenter

_test_dir = Path(__file__).parent


@fixture()
def image() -> np.ndarray:
    return iio.imread(_test_dir.joinpath('test-image.tif'))


@fixture()
def mask() -> np.ndarray:
    img = iio.imread(_test_dir.joinpath('test-image-mask.tif'))
    return np.array(img)


def test_download(tmpdir):
    from rtdefects import segmentation
    orig = segmentation.model_dir
    try:
        segmentation.model_dir = tmpdir
        download_model('README.md')
        assert (tmpdir / 'README.md').read_text('ascii').startswith('#')
    finally:
        segmentation.model_dir = orig


@mark.parametrize(
    'segmenter',
    [TFSegmenter(), PyTorchSegmenter('voids_segmentation_091321.pth'), PyTorchSegmenter('voids_segmentation_030323.pth'),
     PyTorchSegmenter('small_voids_031023.pth'), PyTorchSegmenter()]
)
def test_run(image, segmenter):
    image = segmenter.transform_standard_image(image)
    assert isinstance(image, np.ndarray)
    output = segmenter.perform_segmentation(image)
    output = np.squeeze(output)
    assert output.shape == (1024, 1024)
    assert (output > 0.5).mean() < 0.1


def test_touch_sides(mask):
    """Test whether we detect if voids touch the sides"""
    mask = mask > 0.5

    # Test without an edge buffer
    output, _ = analyze_defects(mask, edge_buffer=0)
    assert sum(output['touches_side']) == 3  # There are three that touch the edge

    # Test with
    output, _ = analyze_defects(mask)
    assert sum(output['touches_side']) > 3  # There are three that touch the edge
