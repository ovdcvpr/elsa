from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import *

import cv2
import numpy as np
from PIL import Image

from elsa.files.files import Files
from elsa.predict.gdino import loader
from elsa.predict.ovdino.errors import DetectronImportError


class ImageLoader(
    loader.ImageLoader,
):
    @property
    def lists(self) -> Iterator[list[np.ndarray]]:
        """Each iteration yields a list of Images"""
        try:
            from detectron2.data.detection_utils import read_image
        except ImportError as e:
            raise DetectronImportError from e

        shapes_files: Iterator[tuple[tuple, Files]] = (
            (shape, files)
            for shape, list_files
            in self.shape2list_files.items()
            for files in list_files
        )
        with ThreadPoolExecutor() as threads:
            it = (
                threads.map(read_image, files.path.values)
                for shape, files in shapes_files
            )
            # preemptive loading of the next image
            prev = next(it)
            while True:
                try:
                    curr = next(it)
                except StopIteration:
                    break
                yield list(prev)
                prev = curr
            yield list(prev)

    # def __iter__(self) -> Iterator[tuple[Tensor, Files]]:
    def __iter__(self) -> Iterator[tuple[
        np.ndarray,
        Image,
    ]]:
        """Each iteration yields an (N, 3, H, W) tensor and the files"""
        it_files = (
            files
            for shape, files
            in self.shape2list_files.items()
            for files in files
        )
        lists = self.lists
        it = zip(it_files, lists)
        for files, images in it:
            images = np.stack(images)
            yield images, files
