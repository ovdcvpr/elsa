from __future__ import annotations

from functools import *
from typing import *

import torch
import torchvision.transforms as T
from tensorflow import Tensor

from elsa.files.files import Files
from elsa.predict.gdino import loader


class ImageLoader(loader.ImageLoader):
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
                transform(image.convert("RGB"))
                .cuda()
                for image in images
            ]
            images = torch.stack(images).cuda()
            yield images, files
