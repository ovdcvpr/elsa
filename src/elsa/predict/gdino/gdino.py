from __future__ import annotations

import magicpandas as magic
from elsa.predict.gdino.batched import Batched

try:
    import open_groundingdino
    import MultiScaleDeformableAttention
except ImportError as e:
    msg = """
    git clone https://github.com/ovdcvpr/Open-GroundingDino.git 
    pip install torch torchvision torchaudio
    cd ./Open-GroundingDino/
    pip install -r requirements.txt 
    cd open_groundingdino/models/GroundingDINO/ops
    python setup.py build install
    python test.py
    cd ../../../../../
    
    """
    raise ImportError(msg) from e


class GDino(magic.Magic):
    @Batched
    def batched(self):
        ...


    def model(self, config, checkpoint, *args, **kwargs):
        from elsa.predict.gdino.model import GroundingDINO
        from elsa.predict.gdino.model import Result
        model = (
            GroundingDINO
            .from_elsa(config, checkpoint)
            .cuda()
        )
        return model
