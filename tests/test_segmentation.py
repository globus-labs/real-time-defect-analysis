"""Test the segmentation function"""
from pytest import mark
import numpy as np

from rtdefects.segmentation.detectron2 import Detectron2Segmenter
from rtdefects.segmentation.pytorch import PyTorchSemanticSegmenter
from rtdefects.segmentation import download_model
from rtdefects.segmentation.tf import TFSegmenter

# Models based on PyTorch segmentation
pytorch_segment = [
    PyTorchSemanticSegmenter('voids_segmentation_091321.pth'), PyTorchSemanticSegmenter('voids_segmentation_030323.pth'),
    PyTorchSemanticSegmenter('small_voids_031023.pth'), PyTorchSemanticSegmenter()
]
detectron2_segment = [
    Detectron2Segmenter(), Detectron2Segmenter('detectron2-dislocations-23Feb24')
]


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
    [TFSegmenter(), *pytorch_segment, *detectron2_segment]
)
def test_run(image, segmenter):
    image = segmenter.transform_standard_image(image)
    assert isinstance(image, np.ndarray)
    image = segmenter.perform_segmentation(image)

    # Make sure the image is the right shape and has some labelled regions
    assert image.shape[1:] == (1024, 1024)
    assert np.any(image, axis=(1, 2)).all()  # All instances must be at least one pixel

