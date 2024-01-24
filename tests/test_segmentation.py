"""Test the segmentation function"""
from pytest import mark
import numpy as np

from rtdefects.segmentation.pytorch import PyTorchSegmenter
from rtdefects.segmentation import download_model
from rtdefects.segmentation.tf import TFSegmenter


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
