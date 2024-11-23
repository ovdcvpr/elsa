from __future__ import annotations
from __future__ import annotations

from functools import *

import magicpandas as magic
from elsa.predict.owl.magic import Owl
from elsa.predict.owlv2.batched import Batched

if False:
    # processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    # model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection


class OwlV2(
    Owl
):
    @cached_property
    def processor(self) -> Owlv2Processor:
        try:
            from transformers import Owlv2Processor
        except ImportError as e:
            msg = f'Please use `pip install "transformers>=4.22\"n; {e}'
            raise ImportError(msg) from e
        processor = (
            Owlv2Processor
            .from_pretrained("google/owlv2-base-patch16-ensemble")
        )
        return processor

    @cached_property
    def model(self) -> Owlv2ForObjectDetection:
        try:
            from transformers import Owlv2ForObjectDetection
        except ImportError as e:
            msg = f"Please use `pip install transformers>=4.22\n; {e}"
            raise ImportError(msg) from e

        model = (
            Owlv2ForObjectDetection
            .from_pretrained("google/owlv2-base-patch16-ensemble")
        )
        # noinspection PyTypeChecker
        model = model.to('cuda')
        return model

    @Batched
    @magic.portal(Batched.__call__)
    def batched(self, ) -> Batched:
        ...
