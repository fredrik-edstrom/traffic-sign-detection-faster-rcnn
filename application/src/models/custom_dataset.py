"""
The CustomDataset class creates a dataset with images and associated annotation-file (xml-format)
"""

import glob as glob

import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as et

from application.src.config import TRAIN_DIR, RESIZE_TO, VALID_DIR, CLASSES, BATCH_SIZE
from application.src.utils.transform import get_train_transform, get_valid_transform
from application.src.utils import collate_fn


class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None) -> None:
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        self.image_paths = glob.glob(f'{self.dir_path}/*.jpg')
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, i: int) -> tuple:
        image_name = self.all_images[i]
        image_path = os.path.join(self.dir_path, image_name)

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        image = Image.open(image_path)
        image_width, image_height = image.size[1], image.size[0]

        boxes, labels = [], []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # left corner x-coordinates
            x_min = int(member.find('bndbox').find('xmin').text)
            # right corner x-coordinates
            x_max = int(member.find('bndbox').find('xmax').text)
            # left corner y-coordinates
            y_min = int(member.find('bndbox').find('ymin').text)
            # right corner y-coordinates
            y_max = int(member.find('bndbox').find('ymax').text)

            x_min = (x_min / image_width) * self.width
            x_max = (x_max / image_width) * self.width
            y_min = (y_min / image_height) * self.height
            y_max = (y_max / image_height) * self.height

            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([i])
        }

        if self.transforms:
            sample = self.transforms(
                image=np.array(image) / 255.0,
                bboxes=target['boxes'],
                labels=labels
            )
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        image = image.swapaxes(0, 1)
        image = image.swapaxes(1, 2)

        return image, target

    def __len__(self) -> int:
        return len(self.all_images)


# prepare datasets and data loaders
train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
