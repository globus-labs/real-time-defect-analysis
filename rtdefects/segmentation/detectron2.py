"""Interface for Detectron2-based models"""
from functools import cached_property
from tarfile import TarFile

import numpy as np
import torch.cuda
from fvcore.common.config import CfgNode
from detectron2.engine import DefaultPredictor

from . import BaseSegmenter, model_dir, download_model


class Detectron2Segmenter(BaseSegmenter):
    """Interface for detectron2-based instance segmentation models

    Args:
        model_name: Name of the model
    """

    def __init__(
            self,
            model_name: str = 'detectron2-dislocations-25Jan24',
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dir = model_dir / model_name

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        return np.repeat(image_data[:, :, None], 3, axis=2)

    @cached_property
    def model(self) -> DefaultPredictor:
        """Underlying Detectron model"""

        # Download and unpack the model if needed
        if not self.model_dir.is_dir():
            download_model(self.model_name + '.tar.xz')
            tar_path = model_dir / (self.model_name + ".tar.xz")
            with TarFile.open(tar_path, 'r:xz') as tar:
                tar.extractall(model_dir)
            tar_path.unlink()

        # Start by loading the model from disk
        cfg_path = self.model_dir / 'config.yaml'
        if not cfg_path.is_file():
            raise ValueError(f'No model configuration found at: {cfg_path}')
        cfg = CfgNode.load_cfg(cfg_path.read_text())

        # Set to use CUDA if is available
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Update the path to the model in that configuration
        cfg.MODEL.WEIGHTS = str(self.model_dir / 'model_final.pth')
        return DefaultPredictor(cfg)

    def perform_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        # Run inference
        outputs = self.model(image_data)
        inst = outputs['instances']

        # Combine them to a single image
        masks_per_inst = inst.pred_masks.cpu().numpy()  # Shape: <instance, x, y>
        inst_id = np.arange(masks_per_inst.shape[0], dtype=np.uint8) + 1
        labeled_mask = np.array(masks_per_inst, dtype=np.uint8) * inst_id[:, None, None]
        return labeled_mask.sum(axis=0)
