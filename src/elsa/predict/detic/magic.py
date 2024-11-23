from __future__ import annotations
import requests

import os

import magicpandas as magic
from elsa.predict.detic.batched import Batched
from elsa.predict.detic.errors import DeticImportError, DetectronImportError, CenterNetImportError
from elsa.resource import Resource

if False:
    import detectron2.utils.comm as comm
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.engine import default_setup
    from detectron2.utils.logger import setup_logger
    from third_party.CenterNet2.centernet.config import add_centernet_config
    from detic.config import add_detic_config
    from detectron2.engine.defaults import DefaultPredictor
    from elsa.predict.detic.predictor import BatchPredictor

"""
Setup for Detectron2 with Detic on the Elsa environment.

Steps:
1. Clone the Elsa environment to set up the necessary dependencies.

2. Detectron2 Installation:
   - Instead of following the standard Detic guide, install Detectron2 using:
     ```
     python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
     ```

3. Detic Installation:
   - Clone the Detic repository with its submodules:
     ```
     git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
     cd Detic
     pip install -r requirements.txt
     ```
   - Note: Most dependencies should already be installed through the Elsa environment setup.

4. Adjust CenterNet Import Path:
   - Update `modeling/backbone/swintransformer.py` to use the correct import path by adding the prefix `Detic.third_party.CenterNet2` to the CenterNet import.

5. Download Configuration and Weights:
   - Place the configuration and model weights in the inference folder by running:
     ```
     wget https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth
     ```

This setup prepares Detectron2 and Detic for inference on the Elsa environment with customized configurations and paths.
"""


class Magic(
    Resource,
):
    @staticmethod
    def setup(args):
        """
        Create configs and perform basic setups.
        """
        try:
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2.engine import default_setup
            from detectron2.utils.logger import setup_logger
            import detectron2.utils.comm as comm
        except ImportError as e:
            raise DetectronImportError() from e
        try:
            from third_party.CenterNet2.centernet.config import add_centernet_config
            from detic.config import add_detic_config
        except ImportError as e:
            raise DeticImportError() from e
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(args["config_file"])
        if '/auto' in cfg.OUTPUT_DIR:
            file_name = os.path.basename(args["config_file"])[:-5]
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        cfg.freeze()
        default_setup(cfg, args)
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detic")
        return cfg

    @magic.cached.static.property
    def metadata(self):
        metadata = MetadataCatalog.get("__unused")
        metadata.thing_classes = self.vocabulary
        return metadata

    @magic.cached.static.property
    def predictor(self) -> BatchPredictor:
        # try:
        #     import centernet
        # except ImportError as e:
        #     raise CenterNetImportError() from e
        try:
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.utils.visualizer import ColorMode, Visualizer
            import detectron2.utils.comm as comm
            from detectron2.config import get_cfg
            from detectron2.data import (MetadataCatalog)
            from detectron2.engine import default_setup
            from detectron2.utils.logger import setup_logger
            from torch.nn import functional as F
            from detectron2.engine.defaults import DefaultPredictor
            from elsa.predict.detic.predictor import BatchPredictor

        except ImportError as e:
            raise DetectronImportError() from e
        try:
            from third_party.CenterNet2.centernet.config import add_centernet_config
            from detic.modeling.text.text_encoder import build_text_encoder
        except ImportError as e:
            raise DeticImportError() from e
        try:
            from detic.config import add_detic_config
        except ImportError as e:
            raise CenterNetImportError() from e

        # __file__
        config_file = os.path.join(
            __file__,
            '../',
            "Detic_OVCOCO_CLIP_R50_1x_max-size_caption.yaml"
        )
        config_file = os.path.abspath(config_file)
        args = {
            "config_file": config_file,
            "vocabulary": "custom",
            "custom_vocabulary": "",  # our classes
            "pred_all_class": False,
            "confidence-threshold": 0.5,
            "cpu": False,
        }
        cfg = self.setup(args)
        # predictor = DefaultPredictor(cfg)  # loads already the checkpoints using the cfg file
        predictor = BatchPredictor(cfg)

        return predictor

    @property
    def weights(self) -> str:
        result = os.path.join(
            '../',
            "Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"
        )
        result = os.path.abspath(result)

        if not os.path.exists(result):
            msg = f'Downloading weights to {result}'
            self.logger.info(msg)
            # URL to download the file
            url = "https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"

            # Download the file and save it to the specified path
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for download errors

            with open(result, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return result



    @Batched
    def batched(self):
        ...
