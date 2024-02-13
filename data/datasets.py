from torchvision.transforms import transforms
import time
from data.training_dataset import TrainingDataset
from data.test_dataset import TestDataset
from data.utils import get_transform
from utils.htr_logging import get_logger
from torch.utils.data import ConcatDataset
from pathlib import Path

logger = get_logger(__file__)


def make_train_dataset(config: dict):
    train_data_path = config['train_data_path']
    patch_size = config['train_patch_size']
    load_data = config['load_data']

    logger.info(f"Train path: \"{train_data_path}\" with patch size {patch_size} and load_data={load_data}")

    transform = get_transform(output_size=patch_size)

    logger.info(f"Loading train datasets...")
    time_start = time.time()
    datasets = []
    for i, path in enumerate(train_data_path):
        logger.info(f"[{i+1}/{len(train_data_path)}] Loading train dataset from \"{path}\"")
        data_path = Path(path) / 'train' if (Path(path) / 'train').exists() else Path(path)
        datasets.append(
            TrainingDataset(
                data_path=data_path,
                split_size=patch_size,
                patch_size=config['train_patch_size_raw'],
                transform=transform,
                load_data=load_data))

    logger.info(f"Loading train datasets took {time.time() - time_start:.2f} seconds")
    train_dataset = ConcatDataset(datasets)
    logger.info(f"Training set has {len(train_dataset)} instances")

    return train_dataset


def make_val_dataset(config: dict):
    val_data_path = config['eval_data_path']
    stride = config['test_stride']
    patch_size = config['eval_patch_size']
    load_data = config['load_data']

    transform = transforms.Compose([transforms.ToTensor()])

    logger.info(f"Loading validation datasets...")
    time_start = time.time()
    datasets = []
    for i, path in enumerate(val_data_path):
        logger.info(f"[{i}/{len(val_data_path)}] Loading validation dataset from \"{path}\"")
        datasets.append(
            TestDataset(
                data_path=Path(path),
                patch_size=patch_size,
                stride=stride,
                transform=transform,
                load_data=load_data
            )
        )

    logger.info(f"Loading validation datasets took {time.time() - time_start:.2f} seconds")
    validation_dataset = ConcatDataset(datasets)
    logger.info(f"Validation set has {len(validation_dataset)} instances")

    return validation_dataset


def make_test_dataset(config: dict):
    test_data_path = config['test_data_path']
    patch_size = config['test_patch_size']
    stride = config['test_stride']
    load_data = config['load_data']

    transform = transforms.Compose([transforms.ToTensor()])

    logger.info(f"Loading test datasets...")
    time_start = time.time()
    datasets = []

    for path in test_data_path:
        datasets.append(
            TestDataset(
                data_path=path,
                patch_size=patch_size,
                stride=stride,
                transform=transform,
                load_data=load_data))
        logger.info(f'Loaded test dataset from {path} with {len(datasets[-1])} instances.')

    logger.info(f"Loading test datasets took {time.time() - time_start:.2f} seconds")

    test_dataset = ConcatDataset(datasets)

    logger.info(f"Test set has {len(test_dataset)} instances")
    return test_dataset
