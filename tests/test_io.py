from pathlib import Path

from imageio import v3 as iio
from pytest import mark
import numpy as np

from rtdefects.io import load_file, read_then_encode, unpack_video, encode_as_tiff

_my_path = Path(__file__).parent


@mark.parametrize(
    'filename', ['test-image.tif', 'test-image.dm4']
)
def test_load(filename: str):
    file = _my_path.joinpath(filename)
    data = load_file(file)
    assert data.ndim == 2
    assert data.dtype == np.float32


def test_transmit():
    message = read_then_encode(_my_path.joinpath('test-image.tif'))
    new_data = iio.imread(message, extension='.tiff')
    assert new_data.dtype == np.uint8


def test_unpack_tiff(tmpdir):
    # Write the multi-image tiff
    tmpdir = Path(tmpdir)
    multiimg_path = tmpdir / 'multi-image.tif'
    frame = iio.imread(_my_path / 'test-image.tif')
    iio.imwrite(multiimg_path, [frame, frame])

    # Make sure we can unpack it
    out_dir = tmpdir / 'frames'
    out_dir.mkdir()
    count = unpack_video(multiimg_path, out_dir)
    assert count == 2
    assert len(list(out_dir.glob("*.tiff"))) == 2


def test_transmit_labelled_image():
    test_img = np.random.random_integers(0, 127, size=(4, 128, 128))
    msg = encode_as_tiff(test_img)
    transmitted_img = iio.imread(msg, extension='.tiff')
    assert transmitted_img.shape == (4, 128, 128)
    assert np.equal(test_img, transmitted_img).all()
