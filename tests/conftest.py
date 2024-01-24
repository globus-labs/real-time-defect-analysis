from pathlib import Path

from pytest import fixture
from imageio import v3 as iio
import numpy as np

_test_dir = Path(__file__).parent


@fixture()
def image() -> np.ndarray:
    return iio.imread(_test_dir.joinpath('test-image.tif'))


@fixture()
def mask() -> np.ndarray:
    img = iio.imread(_test_dir.joinpath('test-image-mask.tif'))
    return np.array(img)
