"""Utilities related to loading and transmitting into standard formats

Our machine learning models expect data as a floating point between 0 and 1.
For bandwidth purposes, we transmit data as 8-bit grayscale TIFFs.
"""

from pathlib import Path
from io import BytesIO

from hyperspy import io as hsio
from skimage import color
import numpy as np
import imageio


def load_file(path: Path) -> np.ndarray:
    """Load a file from disk into a float32 numpy array with values ranging between 0-1.

    Args:
        path: Path to the file of interest
    Returns:
        Data as a standard ndarray
    """

    # Step 1: attempt to read it with imageio
    load_functions = [
        imageio.imread,
        lambda x: hsio.load(x).data
    ]
    data = None
    for function in load_functions:
        try:
            data: np.ndarray = function(path)
        except Exception:
            continue
    if data is None:
        raise ValueError(f'Failed to load image from {path}')

    # Standardize the format
    data = np.array(data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 3:
        data = color.rgb2gray(data)
    data = (data - data.min()) / (data.max() - data.min())
    return data


def encode_as_tiff(data: np.ndarray, compress_type: int = 5) -> bytes:
    """Encode an image as an 8-bit grayscale TIFF, our desired file format

    Args:
        data: Data to be encoded, should be a float array with range 0-1 or already in uint8
        compress_type: Compression algorithm from tifffile
    Returns:
        TIFF image as a byte array
    """

    # Convert mask to a uint8-compatible image
    data = np.squeeze(data)
    if data.dtype.kind not in ['i', 'u']:
        assert np.logical_and(data >= 0, data <= 1).all(), "Image values must be between 0 and 1"
        data = np.array(data * 255, dtype=np.uint8)

    # Convert mask to a TIFF-encoded image
    output_img = BytesIO()
    writer = imageio.get_writer(output_img, format='tiff', mode='i' if data.ndim == 2 else 'I')
    writer.append_data(data, meta={'compression': compress_type})
    return output_img.getvalue()


def read_then_encode(path: Path, compress_level: int = 5) -> bytes:
    """Read an image from disk and return it encoded in the standard TIFF format (8-bit integer)

    Args:
        path: Path to the image file
        compress_level: Lossless compression level, between 0 (no compression) and 9 (maximum)
    Returns:
        TIFF image as a byte array
    """

    data = load_file(path)
    return encode_as_tiff(data, compress_level)


def unpack_video(path: Path | str, output_dir: Path) -> int:
    """Unpack frames from a video and write them to an output directory

    Args:
        path: Path to the video file
        output_dir: Path in which to write frames
    Returns:
        Number of frames
    """

    # Ensure the users give Path objects
    path = Path(path)
    output_dir = Path(output_dir)

    # Unpack
    filename = path.name.lower()
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        img_stack = imageio.mimread(path)
        count = 0
        for img in img_stack:
            imageio.imwrite(output_dir / f'frame-{count:04d}.tiff', img)
            count += 1
        return count
