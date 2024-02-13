from pathlib import Path
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):

    def __init__(self, data_path, patch_size=256, stride=256, transform=None, load_data=True):
        super(TestDataset, self).__init__()

        self.imgs = list((Path(data_path) / 'imgs').glob('*'))
        self.gt_imgs = [img_path.parent.parent / 'gt_imgs' / img_path.name for img_path in self.imgs]

        self.load_data = load_data
        self.imgs_path = self.imgs
        self.gt_imgs_path = self.gt_imgs
        if self.load_data:
            self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
            self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.load_data:
            sample = self.imgs[index]
            gt_sample = self.gt_imgs[index]
        else:
            sample = Image.open(self.imgs[index]).convert("RGB")
            gt_sample = Image.open(self.gt_imgs[index]).convert("L")

        padding_bottom = ((sample.height // self.patch_size) + 1) * self.patch_size - sample.height
        padding_right = ((sample.width // self.patch_size) + 1) * self.patch_size - sample.width

        tensor_padding = functional.to_tensor(sample).unsqueeze(0)
        batch, channels, _, _ = tensor_padding.shape

        tensor_padding = functional.pad(img=tensor_padding, padding=[0, 0, padding_right, padding_bottom], fill=1)
        patches = tensor_padding.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        num_rows = patches.shape[3]
        patches = patches.reshape(batch, channels, -1, self.patch_size, self.patch_size)

        if self.transform:
            sample = self.transform(sample)
            gt_sample = self.transform(gt_sample)

        item = {
            'image_name': str(self.imgs_path[index]),
            'sample': sample,
            'num_rows': num_rows,
            'samples_patches': patches,
            'gt_sample': gt_sample
        }

        return item


class FolderDataset(TestDataset):
    def __init__(self, data_path, patch_size=256, overlap=True, transform=None, load_data=True):
        super(TestDataset, self).__init__()

        self.imgs = list((Path(data_path) / 'imgs').glob('*'))
        self.data_path = data_path
        self.gt_imgs = self.imgs

        self.imgs_path = self.imgs
        self.gt_imgs_path = self.gt_imgs
        if load_data:
            self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
            self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.patch_size = patch_size
        self.stride = patch_size // 2 if overlap else patch_size
        self.transform = transform
        self.load_data = load_data
