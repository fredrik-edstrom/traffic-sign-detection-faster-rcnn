"""
This file contains our final neural network model, we use a Faster RCNN-model
We used it with the Linköping University dataset for our final training
"""

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from application.src.config import CLASSES, DEVICE, NUM_CLASSES


class RCNNModel:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        self.model.to(DEVICE)

    def verify(self, image, target) -> tuple[list, list, list[int], list]:
        """
        Verify the input image against its target labels and bounding boxes
        :param image: PIL image; input image
        :param target: dict; dictionary with target labels etc.
        :return: tuple[list, list, list[int], list]; correct labels, predicted labels, scores and bounding boxes
        """
        with torch.no_grad():
            predictions = self.model(image)[0]

            correct_labels = [CLASSES[label] for label in target['labels']]
            predicted_labels = [CLASSES[label] for label in predictions['labels']]
            scores = [int(score * 100) for score in predictions['scores']]
            boxes = [box for box in predictions['boxes']]

        return correct_labels, predicted_labels, scores, boxes

    def predict(self, image) -> tuple[list, list[int], list]:
        """
        Predict traffic signs in an image
        :param image: PIL image; input image
        :return: tuple[list, list[int], list]; predicted labels, scores and bounding boxes
        """
        with torch.no_grad():
            predictions = self.model(image)[0]

            predicted_labels = [CLASSES[label] for label in predictions['labels']]
            scores = [int(score * 100) for score in predictions['scores']]
            boxes = [box for box in predictions['boxes']]

        return predicted_labels, scores, boxes


def load_model() -> RCNNModel:
    """
    Load the trained RCNN-model from .pth-file
    :return: RCNNModel obj
    """
    from application.src.config import DEVICE, MODEL_NAME, OUT_DIR

    model_path = str(OUT_DIR) + '/' + str(MODEL_NAME) + '.pth'

    nm = RCNNModel()
    nm.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    nm.model.eval()

    return nm
