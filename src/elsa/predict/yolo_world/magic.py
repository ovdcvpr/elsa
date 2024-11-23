from __future__ import annotations
import requests

import os.path

import magicpandas as magic
from elsa.resource import Resource
from elsa.predict.yolo_world.singular import Singular
from elsa.predict.yolo_world.errors import YoloImportError


class Magic(
    Resource,
):

    @Singular
    @magic.portal(Singular.__call__)
    def single(self):
        ...

    @property
    def runner(self):
        try:
            from mmengine.config import Config
            from mmengine.dataset import Compose
            from mmengine.runner import Runner
            from mmengine.runner.amp import autocast
            from mmyolo.registry import RUNNERS
            from torchvision.ops import nms
        except ImportError as e:
            raise YoloImportError from e
        path = os.path.join(
            __file__,
            '../',
            '3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py'
        )
        path = os.path.abspath(path)
        cfg = Config.fromfile(path)
        cfg.work_dir = "."

        path = "yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

        if not os.path.exists(path):
            msg = f'Downloading weights to {path}'
            self.logger.info(msg)
            url = "https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"

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

        cfg.load_from = path
        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        runner.pipeline = Compose(pipeline)



