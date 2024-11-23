from __future__ import annotations

from functools import *

import magicpandas as magic
from elsa.predict.owl.batched import Batched

if False:
    from transformers import OwlViTForObjectDetection, OwlViTProcessor


class Owl(
    magic.Magic,
):
    @cached_property
    def processor(self) -> OwlViTProcessor:
        try:
            from transformers import OwlViTProcessor
        except ImportError as e:
            # msg = f'Please use `pip install "transformers>=4.22"`\"n; {e}'
            msg = f"""
            Please use `
            pip install "transformers>=4.22"
            `; {e}"""
            raise ImportError(msg) from e
        processor = (
            OwlViTProcessor
            .from_pretrained("google/owlvit-base-patch32")
        )
        return processor

    @cached_property
    def model(self) -> OwlViTForObjectDetection:
        try:
            from transformers import OwlViTForObjectDetection
        except ImportError as e:
            msg = f"Please use `pip install transformers>=4.22\n; {e}"
            raise ImportError(msg) from e

        model = (
            OwlViTForObjectDetection
            .from_pretrained("google/owlvit-base-patch32")
        )
        # noinspection PyTypeChecker
        model = model.to('cuda')
        return model

    @Batched
    @magic.portal(Batched.__call__)
    def batched(self, ) -> Batched:
        ...
