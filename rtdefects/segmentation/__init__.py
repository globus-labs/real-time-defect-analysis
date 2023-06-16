"""Functions related to segmentation and analysis of microscopy images"""
from pathlib import Path

import numpy as np
import requests

model_dir = Path(__file__).parent.joinpath('files')


def download_model(name: str):
    """Download a model to local storage

    Args:
        Name of the model
    """
    my_path = model_dir / name
    with requests.get(f"https://g-29c18.fd635.8443.data.globus.org/ivem/models/{name}", stream=True) as r:
        r.raise_for_status()
        with open(my_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


# TODO (wardlt): Support segmentation methods besides semantic segmentation
class BaseSegmenter:
    """Base class for implementations of a segmentation tool

    Implementations must provide a function for reshaping from the format we use
    to transmit images (unit8-based grayscale) into whatever is expected by this specific model,
    and a function that performs the segmentation and returns a boolean array mask.
    """

    def transform_standard_image(self, image_data: np.ndarray) -> np.ndarray:
        """Transform an image into a format compatible with the model

        Args:
            image_data: Image in the as-transmitted format: unit8 grayscale
        Returns:
            Image in whatever form needed by the model
        """
        raise NotImplementedError

    def perform_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Perform the image segmentation

        Args:
            image_data: Images to be segmented.
        Returns:
            Image segmentation mask as a boolean array
        """
        raise NotImplementedError
