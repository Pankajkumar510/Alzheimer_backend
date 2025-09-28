from typing import Tuple

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _GaussianNoise:
    def __init__(self, sigma=0.01, p=0.2):
        self.sigma = sigma
        self.p = p
    def __call__(self, img):
        import numpy as np
        import random
        if random.random() > self.p:
            return img
        arr = np.array(img).astype('float32') / 255.0
        noise = np.random.normal(0, self.sigma, arr.shape).astype('float32')
        arr = np.clip(arr + noise, 0.0, 1.0)
        from PIL import Image
        return Image.fromarray((arr * 255).astype('uint8'))


def get_transforms(image_size: int = 224, is_train: bool = True):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            _GaussianNoise(sigma=0.01, p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def create_vit_model(num_classes: int, model_name: str = "vit_base_patch16_224", pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model
