from __future__ import annotations

import magicpandas as magic
from elsa.predict.mdetr.batched import Batched

try:
    import mdetr
except ImportError as e:
    msg = """
    To predict with MDETR, you need to install it first.
     Please run the following command:

    git clone https://github.com/ovdcvpr/mdetr.git 
    cd ./mdetr/
    pip install .
    """
    raise ImportError(msg) from e

class MDETR(magic.Magic):
    @Batched
    @magic.portal(Batched.__call__)
    def batched(self):
        ...

