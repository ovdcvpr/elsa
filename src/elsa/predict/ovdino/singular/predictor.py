import bisect
import multiprocessing as mp
from copy import copy

import atexit
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]
    return predictions


class OVDINODemo(object):
    def __init__(
            self,
            model,
            sam_predictor,
            min_size_test=800,
            max_size_test=1333,
            img_format="RGB",
            metadata_dataset="coco_2017_val",
            instance_mode=ColorMode.IMAGE,
            parallel=False,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = {}

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.sam_predictor = sam_predictor

        self.parallel = parallel
        self.predictor = DefaultPredictor(
            model=model,
            min_size_test=min_size_test,
            max_size_test=max_size_test,
            img_format=img_format,
            metadata_dataset=metadata_dataset,
        )

    def sam_infer_from_instances(self, image, instances):
        self.sam_predictor.set_image(image)
        boxes = instances.pred_boxes.tensor.detach().numpy()
        if boxes.shape[0] == 0:
            return instances

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        # update pred_masks in instances
        # masks shape: (batch_size) x (num_predicted_masks_per_input) x H x W
        if masks.ndim == 3:
            masks = masks[None, :]
        masks = masks.squeeze(1)
        instances.pred_masks = masks

        return instances

    def run_on_image(
            self, image, category_names, threshold=0.5, with_segmentation=False
    ):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # vis_output = None
        predictions = self.predictor(image, category_names)
        predictions = filter_predictions_with_confidence(predictions, threshold)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        # metadata = {"thing_classes": category_names}
        # visualizer = Visualizer(image, metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            # vis_output = visualizer.draw_panoptic_seg_predictions(
            #     panoptic_seg.to(self.cpu_device), segments_info
            # )
        else:
            if "sem_seg" in predictions:
                ...

                # vis_output = visualizer.draw_sem_seg(
                #     predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                # )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                # Add the mask prediction from the box predictions using SAM.
                if self.sam_predictor is not None and with_segmentation:
                    instances = self.sam_infer_from_instances(image.copy(), instances)
                    predictions["instances"] = instances
                # vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions
        # return predictions, vis_output


class DefaultPredictor:
    def __init__(
            self,
            model,
            min_size_test=800,
            max_size_test=1333,
            img_format="RGB",
            metadata_dataset="coco_2017_val",
    ):
        self.model = model
        # self.model.eval()
        self.metadata = MetadataCatalog.get(metadata_dataset)

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(init_checkpoint)

        self.aug = T.ResizeShortestEdge([min_size_test, min_size_test], max_size_test)

        self.input_format = img_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, category_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {
                "image": image,
                "height": height,
                "width": width,
                "category_names": category_names,
            }
            predictions = self.model([inputs])[0]
            return predictions


