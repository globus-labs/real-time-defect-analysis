"""Implementation using PyTorch.Segmentation"""
from hashlib import md5
from typing import Optional
import logging

import segmentation_models_pytorch as smp

import albumentations as albu
from skimage import color
from skimage.transform import resize
import numpy as np
import torch

from rtdefects.segmentation import SemanticSegmenter, model_dir, download_model

logger = logging.getLogger(__name__)

# Storage for the model
_model: Optional[torch.nn.Module] = None
_loaded_model: Optional[str] = None

# Lookup tables for the pre-processor used by different versions of the model
_encoders = {
    'voids_segmentation_091321.pth': 'se_resnext50_32x4d',
    'voids_segmentation_030323.pth': 'efficientnet-b3',
    'small_voids_031023.pth': 'se_resnext50_32x4d',
}


class PyTorchSegmenter(SemanticSegmenter):
    """Interfaces for models based on segmentation_models.pytorch"""

    def __init__(
            self,
            model_name: str = 'small_voids_031023.pth',
    ):
        """
        Args:
            model_name: Name of the model we should use
        """

        # Get the preprocessor and build the preprocessing pipeline
        assert model_name in _encoders, f'No encoder defined for {model_name}. Consult developer'
        preprocessing_fn = smp.encoders.get_preprocessing_fn(_encoders[model_name])

        # Store the path to the model
        self.model_path = model_dir / model_name
        if not self.model_path.is_file():
            logger.info('Downloading model')
            download_model(model_name)
        assert self.model_path.is_file(), 'Download failed'

        # Define the conversion from image to inputs
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        self.preprocess = albu.Compose(_transform)

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        # Convert to RGB
        image: np.ndarray = color.gray2rgb(image_data)

        # Scale to 1024x1024
        if image.shape[:2] != (1024, 1024):
            image = resize(image, output_shape=(1024, 1024), anti_aliasing=True)

        # Perform the preprocessing
        image = self.preprocess(image=image)

        return image['image']

    def _load_model(self, device: str):
        global _model, _loaded_model
        if _model is None or _loaded_model != self.model_path.name:
            # Make sure the model exists
            if not self.model_path.is_file():
                raise ValueError(f'Cannot find the model. No such file: {self.model_path}')

            # Get the model hash to help with reproducibility
            with open(self.model_path, 'rb') as fp:
                hsh = md5()
                while len(line := fp.read(4096 * 1024)) > 0:
                    hsh.update(line)
            logger.info(f'Loading the model from {self.model_path}. MD5 Hash: {hsh.hexdigest()}')

            # Load it
            _model = torch.load(str(self.model_path), map_location=device)
            logger.info('Model loaded.')
            _loaded_model = self.model_path.name
        return _model

    def generate_mask(self, image_data: np.ndarray) -> np.ndarray:
        # Determine the device at runtime
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self._load_model(device)

        # Push the image to device
        x_tensor = torch.from_numpy(image_data).to(device).unsqueeze(0)

        # Run prediction and get it back from the CPU
        pr_mask = model.predict(x_tensor)
        mask = pr_mask.squeeze().cpu().numpy()

        return mask
