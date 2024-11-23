import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data.transforms import ResizeShortestEdge as T
from detectron2.engine.defaults import DefaultPredictor

class BatchPredictor(DefaultPredictor):
    def __call__(self, images: np.ndarray):
        """
        Args:
            images (np.ndarray): an array of images with shape (N, H, W, C)
                where N is the batch size.

        Returns:
            predictions (list[dict]): a list of predictions, one per image.
        """
        predictions = []

        with torch.no_grad():
            for original_image in images:
                # Preprocess each image
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                image = image.to(self.cfg.MODEL.DEVICE)

                # Prepare input for the model
                inputs = {"image": image, "height": height, "width": width}

                # Run inference on the batch and collect predictions
                prediction = self.model([inputs])[0]
                predictions.append(prediction)

        return predictions
