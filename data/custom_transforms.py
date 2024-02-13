from torchvision import transforms
from torchvision.transforms import functional


class ToTensor(transforms.ToTensor):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        gt = super().__call__(gt)
        return {'image': image, 'gt': gt}


class ColorJitter(transforms.ColorJitter):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        return {'image': image, 'gt': gt}


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        super(RandomCrop, self).__init__(size=size)
        self.size = size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        i, j, h, w = self.get_params(image, output_size=(self.size, self.size))
        image = functional.crop(image, i, j, h, w)
        gt = functional.crop(gt, i, j, h, w)
        return {'image': image, 'gt': gt}


class RandomRotation(transforms.RandomRotation):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        angle = self.get_params(self.degrees)

        image = functional.rotate(image, angle, fill=[255, 255, 255])

        gt = functional.invert(gt)
        gt = functional.rotate(gt, angle)
        gt = functional.invert(gt)
        return {'image': image, 'gt': gt}
