
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from functools import *
from pandas import DataFrame
from tensorflow import Tensor
from typing import *
from elsa.files.files import Files


class ImageLoader:
    def __init__(
            self,
            files: Files,
            batch_size: int,
    ):
        """Depending on batch size map each shape to a list of batches"""
        self.files = files
        self.batch_size = batch_size

        def apply(frame: DataFrame):
            return [
                frame.iloc[i:i + batch_size]
                for i in range(0, len(frame), batch_size)
            ]

        _ = files.height, files.width
        self.shape2list_files: dict[tuple, list[Files]] = (
            files
            .groupby('height width'.split(), sort=False)
            .apply(apply)
            .to_dict()
        )


    @cached_property
    def transform(self):
        """See groundingdino.util.load_image"""
        import open_groundingdino.datasets.transforms as T
        transforms = [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        transform = T.Compose(transforms)
        return transform

    @property
    def lists(self) -> Iterator[list[Image.Image]]:
        """Each iteration yields a list of Images"""
        shapes_files: Iterator[tuple[tuple, Files]] = (
            (shape, files)
            for shape, list_files
            in self.shape2list_files.items()
            for files in list_files
        )
        with ThreadPoolExecutor() as threads:
            it = (
                threads.map(Image.open, files.path.values)
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

    def __iter__(self) -> Iterator[tuple[Tensor, Files]]:
        """Each iteration yields an (N, 3, H, W) tensor and the files"""
        it_files = (
            files
            for shape, files
            in self.shape2list_files.items()
            for files in files
        )
        lists = self.lists
        it = zip(it_files, lists)
        transform = self.transform
        for files, images in it:
            images = [
                transform(image.convert("RGB"), target=None)[0]
                for image in images
            ]
            images = torch.stack(images).cuda()
            yield images, files


