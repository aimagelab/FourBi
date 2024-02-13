import logging
import cv2
import numpy as np
from pathlib import Path


class PatchImage:

    def __init__(self, patch_size: int, overlap_size: int, destination_root: str):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        destination_root = Path(destination_root)
        self.train_folder = destination_root / f'imgs_{patch_size}/'
        self.train_gt_folder = destination_root / f'gt_imgs_{patch_size}/'
        self.train_folder.mkdir(parents=True, exist_ok=True)
        self.train_gt_folder.mkdir(parents=True, exist_ok=True)

        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.number_image = 1
        self.image_name = ""

        logging.info(f"Using Patch size: {self.patch_size} - Overlapping: {self.overlap_size}")

    def create_patches(self, root_original: str):
        logging.info("Start process ...")
        root_original = Path(root_original)
        gt = root_original / 'gt_imgs'
        imgs = root_original / 'imgs'

        path_imgs = list(path_img for path_img in imgs.glob('*') if path_img.suffix in {".png", ".jpg", ".bmp", ".tif"})
        for i, img in enumerate(path_imgs):
            or_img = cv2.imread(str(img))
            gt_img = gt / img.name
            gt_img = gt_img if gt_img.exists() else gt / (img.stem + '.png')
            gt_img = cv2.imread(str(gt_img))
            try:
                self._split_train_images(or_img, gt_img)
            except Exception as e:
                print(f'Error: {e} - {img}')

    def _split_train_images(self, or_img: np.ndarray, gt_img: np.ndarray):
        runtime_size = self.overlap_size
        patch_size = self.patch_size
        for i in range(0, or_img.shape[0], runtime_size):
            for j in range(0, or_img.shape[1], runtime_size):

                if i + patch_size <= or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = or_img[i:i + patch_size, j:j + patch_size, :]
                    gt_patch = gt_img[i:i + patch_size, j:j + patch_size, :]

                elif i + patch_size > or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, :, :] = or_img[i:or_img.shape[0], j:j + patch_size, :]
                    gt_patch[0:or_img.shape[0] - i, :, :] = gt_img[i:or_img.shape[0], j:j + patch_size, :]

                elif i + patch_size <= or_img.shape[0] and j + patch_size > or_img.shape[1]:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[:, 0:or_img.shape[1] - j, :] = or_img[i:i + patch_size, j:or_img.shape[1], :]
                    gt_patch[:, 0:or_img.shape[1] - j, :] = gt_img[i:i + patch_size, j:or_img.shape[1], :]

                else:
                    dg_patch = np.ones((patch_size, patch_size, 3)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = or_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]

                cv2.imwrite(str(self.train_folder / (str(self.number_image) + '.png')), dg_patch)
                cv2.imwrite(str(self.train_gt_folder / (str(self.number_image) + '.png')), gt_patch)
                self.number_image += 1
                print(self.number_image, end='\r')

    def _create_name(self, folder: str, i: int, j: int):
        return folder + self.image_name.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'
