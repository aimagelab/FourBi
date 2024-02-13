import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms, functional
from torchvision.utils import make_grid

import data.custom_transforms as CustomTransform


def get_transform(output_size: int):
    transform_list = [
        CustomTransform.ToTensor(),
        CustomTransform.RandomRotation((-10, 10)),
        CustomTransform.RandomCrop(output_size),
        CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
    ]

    transform = transforms.Compose(transform_list)
    return transform


def reconstruct_ground_truth(patches, original, num_rows, config):
    channels = 1
    batch_size = 1
    patch_size = config['test_patch_size']
    stride = config['test_stride']

    _, _, height, width = original.shape
    width, height = original.shape[-1], original.shape[-2]
    tmp_patches = patches.view(batch_size, channels, -1, num_rows, patch_size, patch_size)
    patch_width, patch_height = tmp_patches.shape[-1], tmp_patches.shape[-2]
    tensor_padded_width = patch_width * tmp_patches.shape[-3] - (patch_width - stride) * (tmp_patches.shape[-3] - 1)
    tensor_padded_height = patch_height * tmp_patches.shape[-4] - (patch_height - stride) * (tmp_patches.shape[-4] - 1)

    padding_up = 0
    padding_left = 0

    if stride == (patch_size // 2):
        patches = tmp_patches

        x_steps = [x + (stride // 2) for x in range(0, tensor_padded_height, stride)]
        x_steps[0], x_steps[-1] = 0, tensor_padded_height
        y_steps = [y + (stride // 2) for y in range(0, tensor_padded_width, stride)]
        y_steps[0], y_steps[-1] = 0, tensor_padded_width

        canvas = torch.zeros(batch_size, channels, tensor_padded_height, tensor_padded_width)
        for j in range(len(x_steps) - 1):
            for i in range(len(y_steps) - 1):
                patch = patches[0, :, j, i, :, :]
                x1_abs, x2_abs = x_steps[j], x_steps[j + 1]
                y1_abs, y2_abs = y_steps[i], y_steps[i + 1]
                x1_rel, x2_rel = x1_abs - (j * stride), x2_abs - (j * stride)
                y1_rel, y2_rel = y1_abs - (i * stride), y2_abs - (i * stride)
                canvas[0, :, x1_abs:x2_abs, y1_abs:y2_abs] = patch[:, x1_rel:x2_rel, y1_rel:y2_rel]
        canvas = functional.crop(canvas, top=padding_up, left=padding_left, height=height, width=width)
        canvas = canvas.to(original.device)
    else:
        tensor = make_grid(patches, nrow=num_rows, padding=0, value_range=(0, 1))
        tensor = functional.rgb_to_grayscale(tensor)
        _, _, height, width = original.shape
        canvas = functional.crop(tensor, top=padding_up, left=padding_left, height=height, width=width)
        canvas = canvas.unsqueeze(0)

    return canvas
