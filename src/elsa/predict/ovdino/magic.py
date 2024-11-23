from __future__ import annotations

import magicpandas as magic
from elsa.predict.ovdino.batched.batched import Batched
from elsa.predict.ovdino.singular.singular import Singular
from elsa.resource import Resource
from elsa.predict.ovdino.errors import DetrixImportError, DetectronImportError

import multiprocessing as mp
import requests
import os



class Magic(
    Resource,
):
    @Batched
    @magic.portal(Batched.__call__)
    def batched(self):
        ...

    @Singular
    @magic.portal(Singular.__call__)
    def singular(self):
        ...
    # @magic.cached.static.property
    @property
    def model(self):
        try:
            from ovdino.demo.predictors import OVDINODemo
            from ovdino.projects.ovdino.configs.models.ovdino_swin_tiny224_bert_base import model
        except ImportError as e:
            raise DetrixImportError() from e
        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import LazyConfig, instantiate
            from detectron2.utils.logger import setup_logger
            from detectron2.data.detection_utils import read_image
        except ImportError as e:
            raise DetectronImportError() from e
        mp.set_start_method("spawn", force=True)
        setup_logger(name="fvcore")

        model = instantiate(model)
        model.to("cuda")

        path = "ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth"
        path = os.path.abspath(path)

        if not os.path.exists(path):
            msg = f'Downloading weights to {path}'
            self.logger.info(msg)
            url = "https://huggingface.co/hao9610/OV-DINO/resolve/main/ovdino_swint_ogc-coco50.2_lvismv40.1_lvis32.9.pth"

            try:
                # Start the download with streaming
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for any HTTP errors

                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info("Download completed successfully.")

            except requests.RequestException as e:
                self.logger.error(f"Error downloading weights: {e}")
                if os.path.exists(path):
                    os.remove(path)  # Remove incomplete file if download fails
                raise

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(path)
        model.eval()

        return model

    # @magic.cached.static.property
    # def predictor(self) -> BatchPredictor:
    #     try:
    #         from elsa.predict.ovdino.predictor import BatchPredictor
    #     except ImportError:
    #         ...

    @magic.cached.static.property
    def demo(self):
        # from elsa.predict.ovdino.predictor import ParallelOVDINODemo
        from elsa.predict.ovdino.predictor import OVDINODemo
        # result = ParallelOVDINODemo(
        result = OVDINODemo(
            model=self.model,
            sam_predictor=None,
            img_format="BGR",
        )
        return result

    @Batched
    @magic.portal(Batched.__call__)
    def batched(self):
        ...
