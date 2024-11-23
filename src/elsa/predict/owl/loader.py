from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torchvision.transforms as T
import torch
from PIL import Image
from functools import *
from pandas import DataFrame
from tensorflow import Tensor
from typing import *
from elsa.files.files import Files
from elsa.predict.gdino import loader


class ImageLoader(
    loader.ImageLoader,
):
    # def __iter__(self) -> Iterator[tuple[Tensor, Files]]:
    def __iter__(self) -> Iterator[tuple[
        torch.Tensor,
        Files,
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
        transform = self.transform
        for files, images in it:
            pils = images
            images = [
                transform(image)
                for image in images
            ]
            images = torch.stack(images).cuda()
            yield images, files, pils

    @cached_property
    def transform(self):
        """See groundingdino.util.load_image"""
        transforms = [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        transform = T.Compose(transforms)
        return transform
