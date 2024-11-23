from __future__ import annotations

import inspect
import os

import magicpandas as magic

config = {
    "predict": {
        "gdino": {
            "config": {
                "redacted": "/home/redacted/projects/elsa/configs/cfg_odvg.py",
                "redacted": "/home/redacted/projects/elsa/configs/cfg_odvg.py"
            },
            "checkpoint": {
                "redacted": "/home/redacted/Downloads/gdinot-coco-ft.pth",
                "redacted": "/home/redacted/projects/elsa/weights/gdinot-coco-ft.pth",
                "redacted": "/home/redacted/projects/elsa/weights/groundingdino_swint_ogc.pth",
            },
            "batch_size": {
                "redacted": 8,
                "redacted": 64,
                "redacted": 128,
            }
        },
        "glip": {
            "config": {

            },
            "checkpoint": {

            },
            "batch_size": {

            }
        },
    },
    "files": {
        "bing": {
            'redacted': '/home/redacted/Downloads/images/',
        },
        "google": {
            "redacted": "/home/redacted/Downloads/yolo/images/",
        }
    }
}


class GDino(magic.Magic):
    @magic.cached.local.property
    def config(self) -> str:
        """
        Grounding DINO config file; should be located under
        Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py
        """
        import open_groundingdino.config.cfg_odvg as config
        result = inspect.getfile(config)
        return result

    @magic.cached.local.property
    def checkpoint(self) -> str:
        """
        GroundingDINO checkpoint file available for download at
        https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#luggage-checkpoints
        """
        result = os.path.join(
            os.getcwd(),
            'groundingdino_swint_ogc.pth'
        )
        msg = (
            f"No checkpoint file specified; defaulting to {result}. "
            f"This is the current working directory."
        )
        self.logger.info(msg)
        return result

    @magic.cached.local.property
    def batch_size(self) -> int:
        """Maximum number of images to inference on in parallel at a time"""
        # self.logger.info("No batch size specified; ")
        result = 1
        msg = (
            f"No batch size specified; defaulting to {result}. "
            f"This may be very slow."
        )
        self.logger.info(msg)
        return result


class Glip(GDino):
    @magic.cached.local.property
    def config(self) -> str:
        """
        Grounding DINO config file; should be located under
        Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        result = os.path.join(current_dir, 'predict', 'glip', 'glip_Swin_L.yaml')
        return result

    @magic.cached.local.property
    def checkpoint(self) -> str:
        """
        GroundingDINO checkpoint file available for download at
        https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#luggage-checkpoints
        """
        result = os.path.join(
            os.getcwd(),
            'glip_large_model.pth'
        )
        msg = (
            f"No checkpoint file specified; defaulting to {result}. "
            f"This is the current working directory."
        )
        self.logger.info(msg)
        return result

    @magic.cached.local.property
    def batch_size(self) -> int:
        """Maximum number of images to inference on in parallel at a time"""
        # self.logger.info("No batch size specified; ")
        result = 1
        msg = (
            f"No batch size specified; defaulting to {result}. "
            f"This may be very slow."
        )
        self.logger.info(msg)
        return result


class Predict(magic.Magic.Blank):
    gdino = GDino()
    glip = Glip()


class Files(magic.Magic.Blank):
    @magic.cached.local.property
    def bing(self):
        """Path to Bing SV images"""
        return None

    @magic.cached.local.property
    def google(self):
        """Path to Google SV images"""
        return None

    @magic.cached.sticky.property
    def unified(self) -> tuple:
        """Path to both Bing and Google SV images"""
        result = []
        if self.bing:
            result.append(self.bing)
        if self.google:
            result.append(self.google)
        result = tuple(result)
        return result


class Locals(magic.Locals.Blank):
    predict = Predict()
    files = Files()


local = Locals(config)
