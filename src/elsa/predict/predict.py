from __future__ import annotations
from __future__ import annotations

from typing import *

import magicpandas as magic

if False:
    from elsa.root import Elsa
    import elsa.predict.gdino.gdino
    from elsa.predict.mdetr.magic import MDETR
    from elsa.predict.gdino.batched import Batched
    from elsa.annotation.prompts import Prompts
    from elsa.predict.owl.magic import Owl
    from elsa.predict.owlv2.magic import OwlV2
    from elsa.predict.detic.magic import Magic as Detic
    from elsa.predict.ovdino.magic import Magic as OVDino


def call(self: Predict, *args, **kwargs):
    return self.gdino.batched(*args, **kwargs)


class Predict(
    magic.Magic.Blank
):
    outer: Elsa
    gdino: elsa.predict.gdino.gdino.GDino
    owlv2: elsa.predict.owlv2.magic.OwlV2
    mdetr: elsa.predict.mdetr.magic.MDETR
    owl: elsa.predict.owl.magic.Owl
    detic: elsa.predict.detic.magic.Magic
    ovdino: elsa.predict.ovdino.magic.Magic

    locals().update(__call__=call)

    @magic.delayed
    def gdino(self) -> elsa.predict.gdino.gdino.GDino:
        ...

    @magic.delayed
    def mdetr(self) -> elsa.predict.mdetr.magic.MDETR:
        ...

    @magic.delayed
    def owl(self) -> Owl:
        ...

    @magic.delayed
    def owlv2(self) -> OwlV2:
        ...

    @magic.delayed
    def detic(self) -> Detic:
        ...

    @magic.delayed
    def ovdino(self) -> OVDino:
        ...

    def __call__(
            self,
            outdir: str = None,
            batch_size: int = None,
            force=False,
            synonyms: int = None,
            prompts: int | Collection[bool] | Prompts = None,
            files: int | Collection[bool] = None,
            config=None,
            checkpoint=None,
    ) -> Batched:
        return self.gdino.batched(
            outdir,
            batch_size=batch_size,
            force=force,
            synonyms=synonyms,
            prompts=prompts,
            files=files,
            config=config,
            checkpoint=checkpoint,
        )

    if False:
        __call__ = GDino.batched.__call__
    locals()['__call__'] = call

"""
zbWe currently offer these models. The installation instructions will be
provided when you attempt to access them with the respective code:

GDINO:
    `elsa.predict.gdino`
MDETR:
    `elsa.predict.mdetr`
OWL:
    `elsa.predict.owl`
OWL V2:
    `elsa.predict.owlv2`
DETIC:
    `elsa.predict.detic`
"""