from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class ValidationDataset(Dataset):

    def __init__(self, data_path, split_size=256, patch_size=384, transform=None, load_data=True):
        super(ValidationDataset, self).__init__()
        self.imgs = list(Path(data_path).rglob(f'val_imgs_{split_size}/*'))
        self.gt_imgs = [img_path.parent.parent / ('val_gt_' + img_path.parent.name[4:]) / img_path.name for img_path in
                        self.imgs]

        self.load_data = load_data
        if self.load_data:
            self.imgs = [Image.open(img_path).convert("RGB") for img_path in self.imgs]
            self.gt_imgs = [Image.open(gt_img_path).convert("L") for gt_img_path in self.gt_imgs]

        self.split_size = split_size
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

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        gt_sample = gt_sample.float()

        return sample, gt_sample
