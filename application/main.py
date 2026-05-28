"""
Temporary main file
"""

import os
import random as r
import torch

from PIL import Image
from torchvision.transforms import transforms

from application.src.config import TEST_PATH
from application.src.models.faster_rcnn import RCNNModel
from application.src.utils.presentation import get_relevant_scores, show


def load_model():
    from application.src.config import DEVICE, MODEL_NAME, OUT_DIR

    model_path = str(OUT_DIR) + '/' + str(MODEL_NAME) + '.pth'

    nm = RCNNModel()
    nm.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    nm.model.eval()

    return nm


def print_result(labels: list, scores=None) -> None:
    """
    Print the labels and, if we predict a random image, the accuracy score.
    Only prints the signs that with more than 60% accuracy
    :param labels: list; list with the predicted or actual signs
    :param scores: list[int]; list with accuracy scores, only when predicted. None by default
    :return: None
    """
    if scores:
        [print(f'Sign: {label} - Accuracy: {scores[i]}%') for i, label in enumerate(labels)]
    else:
        [print(f'Sign: {label}') for label in labels]


def main():
    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    filename = r.choice(os.listdir(TEST_PATH))

    if filename[-3:] == 'jpg':
        path = '%s/%s' % (TEST_PATH, filename)
    else:
        filename = filename[:-3] + 'jpg'
        path = '%s/%s' % (TEST_PATH, filename)

    image = Image.open(path)
    prediction_image = transform(image)
    prediction_image = prediction_image.view(1, 3, prediction_image.shape[1], prediction_image.shape[2])

    labels, scores, boxes = model.predict(prediction_image)

    labels, scores, boxes = get_relevant_scores(labels, scores, boxes)

    show(image, labels, boxes, scores)
    print_result(labels, scores)


if __name__ == '__main__':
    main()
