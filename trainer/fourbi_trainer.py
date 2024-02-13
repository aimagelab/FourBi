import errno
import math
import os
import wandb
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import functional

from data.dataloaders import make_train_dataloader, make_valid_dataloader, make_test_dataloader
from data.datasets import make_train_dataset, make_val_dataset, make_test_dataset
from data.utils import reconstruct_ground_truth
from modules.ffc import Fourbi
from trainer.losses import make_criterion
from trainer.optimizers import make_optimizer
from trainer.schedulers import make_lr_scheduler
from trainer.validator import Validator
from utils.htr_logging import get_logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class FourbiTrainingModule:

    def __init__(self, config, device=None, make_loaders=True):
        self.config = config
        self.device = device
        self.checkpoint = None

        if 'resume' in self.config:
            self.checkpoint = torch.load(config['resume'], map_location=device)
            checkpoint_config = self.checkpoint['config'] if 'config' in self.checkpoint else {}
            if 'train_data_path' in checkpoint_config:
                del checkpoint_config['train_data_path']
            if 'eval_data_path' in checkpoint_config:
                del checkpoint_config['eval_data_path']
            if 'test_data_path' in checkpoint_config:
                del checkpoint_config['test_data_path']
            self.config.update(checkpoint_config)

            config = self.config

        if make_loaders:
            self.train_dataset = make_train_dataset(config)
            self.eval_dataset = make_val_dataset(config)
            self.test_dataset = make_test_dataset(config)

            self.train_data_loader = make_train_dataloader(self.train_dataset, config)
            self.eval_data_loader = make_valid_dataloader(self.eval_dataset, config)
            self.test_data_loader = make_test_dataloader(self.test_dataset, config)

        self.model = Fourbi(input_nc=config['input_channels'], output_nc=config['output_channels'],
                            n_downsampling=config['n_downsampling'], init_conv_kwargs=config['init_conv_kwargs'],
                            downsample_conv_kwargs=config['down_sample_conv_kwargs'],
                            resnet_conv_kwargs=config['resnet_conv_kwargs'], n_blocks=config['n_blocks'],
                            unet_layers=config['unet_layers'], )
        config['num_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Training
        self.epoch = 0
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.seed = config['seed']

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model'], strict=True)
            self.epoch = self.checkpoint['epoch'] + 1
            self.best_psnr = self.checkpoint['best_psnr']
            self.learning_rate = self.checkpoint['learning_rate']
            self.load_random_settings()

        self.model = self.model.to(self.device)

        self.optimizer = make_optimizer(self.model, self.learning_rate, config['optimizer'])
        self.lr_scheduler = make_lr_scheduler(config['lr_scheduler'], self.optimizer, config['lr_scheduler_kwargs'],
                                              config['lr_scheduler_warmup'], config)
        self.criterion = make_criterion(losses=config['losses'])

        # Validation
        self.best_epoch = 0
        self.psnr_list = []
        self.best_psnr = 0.
        self.best_precision = 0.
        self.best_recall = 0.

        # Logging
        self.logger = get_logger(FourbiTrainingModule.__name__)

        # Resume
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            if 'lr_scheduler' in self.checkpoint:
                if self.checkpoint['lr_scheduler'] is not None:
                    self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
            self.logger.info(f"Loaded pretrained checkpoint model from \"{config['resume']}\"")

    def load_random_settings(self):
        if 'random_settings' in self.checkpoint:
            set_seed(self.checkpoint['random_settings']['seed'])
            random.setstate(self.checkpoint['random_settings']['random_rng_state'])
            np.random.set_state(self.checkpoint['random_settings']['numpy_rng_state'])
            torch.set_rng_state(self.checkpoint['random_settings']['torch_rng_state'].type(torch.ByteTensor))
            torch.cuda.set_rng_state(self.checkpoint['random_settings']['cuda_rng_state'].type(torch.ByteTensor))

    def _save_checkpoint(self, model_state_dict, root_folder, filename: str):
        random_settings = {'random_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
                           'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                           'seed': self.seed}

        checkpoint = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'learning_rate': self.learning_rate,
            'config': self.config,
            'random_settings': random_settings,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

        if wandb.run is not None:
            checkpoint['wandb_id'] = wandb.run.id

        dir_path = root_folder / f"{filename}.pth"
        torch.save(checkpoint, dir_path)
        return dir_path

    def save_checkpoints(self, filename: str, root_folder=None):
        root_folder = Path(self.config['checkpoint_dir']) if root_folder is None else Path(root_folder)
        root_folder.mkdir(parents=True, exist_ok=True)
        dir_path = self._save_checkpoint(self.model.state_dict(), root_folder, filename)
        self.logger.info(f"Stored checkpoints {dir_path}")

    def eval_item(self, item, validator, threshold):
        image_name = item['image_name'][0]
        sample = item['sample']
        num_rows = item['num_rows'].item()
        samples_patches = item['samples_patches']
        gt_sample = item['gt_sample']

        samples_patches = samples_patches.squeeze(0)
        test = samples_patches.to(self.device)
        gt_test = gt_sample.to(self.device)

        test = test.squeeze(0)
        test = test.permute(1, 0, 2, 3)

        pred = []
        for chunk in torch.split(test, self.config['eval_batch_size']):
            pred.append(self.model(chunk))
        pred = torch.cat(pred)

        pred = reconstruct_ground_truth(pred, gt_test, num_rows=num_rows, config=self.config)
        loss = self.criterion(pred, gt_test)

        if threshold is not None:
            pred = torch.where(pred > threshold, 1., 0.)
            validator.compute(pred, gt_test)

        test = sample.squeeze().detach()
        pred = pred.squeeze().detach()
        gt_test = gt_test.squeeze().detach()
        test_img = functional.to_pil_image(test)
        pred_img = functional.to_pil_image(pred)
        gt_test_img = functional.to_pil_image(gt_test)
        images = {image_name: [test_img, pred_img, gt_test_img]}

        if 'degrees' in item:
            images[image_name].append(item['degrees'])
        if 'max_offset' in item:
            images[image_name].append(item['max_offset'])

        return loss.item(), validator, images

    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_test'], threshold=threshold)

        for i, item in enumerate(self.test_data_loader):
            test_loss_item, validator, images_item = self.eval_item(item, validator, threshold)
            test_loss += test_loss_item
            images.update(images_item)

        avg_loss = test_loss / len(self.test_data_loader)
        avg_metrics = validator.get_metrics()

        self.model.train()
        return avg_metrics, avg_loss, images

    @torch.no_grad()
    def folder_test(self, threshold=None):
        self.model.eval()
        threshold = self.config['threshold'] if threshold is None else threshold

        validator = Validator(apply_threshold=self.config['apply_threshold_to_test'], threshold=threshold)

        for i, item in enumerate(self.test_data_loader):
            _, _, images_item = self.eval_item(item, validator, threshold)
            yield images_item

    @torch.no_grad()
    def validation(self):
        self.model.eval()
        eval_loss = 0.0
        threshold = self.config['threshold']

        images = {}
        validator = Validator(apply_threshold=self.config['apply_threshold_to_eval'], threshold=threshold)

        for item in self.eval_data_loader:
            eval_loss_item, validator, images_item = self.eval_item(item, validator, threshold)
            eval_loss += eval_loss_item
            images.update(images_item)

        avg_loss = eval_loss / len(self.eval_data_loader)
        avg_metrics = validator.get_metrics()

        self.model.train()
        return avg_metrics, avg_loss, images
