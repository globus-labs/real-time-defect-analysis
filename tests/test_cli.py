import shutil
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json

import imageio
from pytest import fixture
from imageio import imread

from rtdefects.cli import _funcx_func, LocalProcessingHandler, main
from rtdefects.segmentation.pytorch import PyTorchSegmenter
from rtdefects.segmentation.tf import TFSegmenter

test_image = Path(__file__).parent.joinpath("test-image.tif")


@fixture()
def multi_image(tmpdir):
    output = Path(tmpdir) / 'image-stack.tiff'
    frame = imageio.imread(test_image)
    imageio.mimwrite(output, [frame] * 4)
    return output


def test_funcx():
    """Test the funcx function"""
    data = test_image.read_bytes()
    mask_bytes, defect_info = _funcx_func(TFSegmenter(), data)
    mask = imread(BytesIO(mask_bytes), format='tiff')
    assert 0 < mask.mean() < 255, "Mask is a single color."
    assert mask.max() == 255
    print(json.dumps(defect_info))


def test_local_reader():
    reader = LocalProcessingHandler(PyTorchSegmenter())
    reader.submit_file(test_image, datetime.now())
    img_path, mask, defect_info, rtt, detect_time = next(reader.iterate_results())


def test_run(multi_image: Path):
    # Run on a video file
    main(['--local', 'run', str(multi_image.absolute())])
    out_dir = multi_image.parent / (multi_image.name + "_run")
    assert out_dir.is_dir()
    assert (out_dir / 'masks').is_dir()
    assert len(list(out_dir.glob("*.tiff"))) == 4
    assert len(list(out_dir.glob("masks/*.tiff"))) == 4

    # Re-run on the directory it produced
    shutil.rmtree(out_dir / 'masks')
    main(['--local', 'run', str(out_dir.absolute())])
    assert (out_dir / 'masks').is_dir()
    assert len(list(out_dir.glob("masks/*.tiff"))) == 4
