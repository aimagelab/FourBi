import argparse
import sys
import time
import uuid
import traceback
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import wandb
from torchvision.transforms import functional

from trainer.fourbi_trainer import FourbiTrainingModule, set_seed
from trainer.validator import Validator
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger

logger = get_logger('main')


def train(config):
    wandb_log = None
    device = config['device']
    trainer = FourbiTrainingModule(config, device=device)

    if config['use_wandb']:  # Configure WandB
        tags = [Path(path).name for path in config['train_data_path']]
        wandb_id = wandb.util.generate_id()
        if trainer.checkpoint is not None and 'wandb_id' in trainer.checkpoint:
            wandb_id = trainer.checkpoint['wandb_id']
        wandb_log = WandbLog(experiment_name=config['experiment_name'], tags=tags, project=config['wandb_project'],
                             entity=config['wandb_entity'], dir=config['wandb_dir'], id=wandb_id)
        wandb_log.setup(config)

    if wandb_log:
        wandb_log.add_watch(trainer.model)

    threshold = config['threshold']
    train_validator = Validator(apply_threshold=config['apply_threshold_to_train'], threshold=threshold)

    try:
        patience = config['patience']
        training_start_time = time.time()

        for epoch_idx, epoch in enumerate(range(trainer.epoch, config['num_epochs'])):
            wandb_logs = dict()
            wandb_logs['lr'] = trainer.optimizer.param_groups[0]['lr']
            trainer.epoch = epoch

            logger.info("Training started") if epoch == 0 else None
            remaining_time = (time.time() - training_start_time) / (epoch_idx + 1) * (config['num_epochs'] - epoch)
            train_eta = str(timedelta(seconds=remaining_time)) if epoch_idx > 0 else "N/A"
            logger.info(f"Epoch [{trainer.epoch}/{trainer.num_epochs}]. Patience: {patience}. ETA: {train_eta}")

            train_loss = 0.0
            trainer.model.train()
            train_validator.reset()

            data_times = []
            train_times = []
            start_data_time = time.time()
            start_epoch_time = time.time()

            for batch_idx, (images, images_gt) in enumerate(trainer.train_data_loader):
                data_times.append(time.time() - start_data_time)
                start_train_time = time.time()
                images, images_gt = images.to(device), images_gt.to(device)

                trainer.optimizer.zero_grad()
                predictions = trainer.model(images)
                loss = trainer.criterion(predictions, images_gt)
                loss.backward()
                trainer.optimizer.step()
                train_loss += loss.item()

                train_times.append(time.time() - start_train_time)

                with torch.no_grad():
                    if batch_idx % config['train_log_every'] == 0:
                        metrics = train_validator.compute(predictions, images_gt)

                        size = batch_idx * len(images)
                        percentage = 100. * size / len(trainer.train_dataset)
                        elapsed_time = time.time() - start_epoch_time
                        if batch_idx > 0:
                            eta = str(timedelta(seconds=(len(trainer.train_dataset) - size) * (elapsed_time / size)))
                        else:
                            eta = "N/A"

                        logger.info(f'[{size:05d}/{len(trainer.train_dataset)}] ({percentage:.2f}%). Train Loss: '
                                    f'{loss.item():.6f}. PSNR: {metrics["psnr"]:0.4f}. Epoch eta: {eta}')

                start_data_time = time.time()

            avg_train_loss = train_loss / len(trainer.train_dataset)
            avg_train_metrics = train_validator.get_metrics()
            train_validator.reset()

            logger.info(f"AVG train loss: {avg_train_loss:0.4f} - AVG training PSNR: {avg_train_metrics['psnr']:0.4f}")

            wandb_logs['train/avg_loss'] = avg_train_loss
            wandb_logs['train/avg_psnr'] = avg_train_metrics['psnr']
            wandb_logs['train/data_time'] = np.array(data_times).mean()
            wandb_logs['train/time_per_iter'] = np.array(train_times).mean()

            original = images[0]
            pred = predictions[0].expand(3, -1, -1)
            output = images_gt[0].expand(3, -1, -1)
            union = torch.cat((original, pred, output), 2)
            wandb_logs['Random Sample'] = wandb.Image(functional.to_pil_image(union), caption=f"Example")

            with torch.no_grad():
                start_eval_time = time.time()
                eval_metrics, eval_loss, _ = trainer.validation()

                wandb_logs['eval/time'] = time.time() - start_eval_time
                wandb_logs['eval/avg_loss'] = eval_loss
                wandb_logs['eval/avg_psnr'] = eval_metrics['psnr']
                wandb_logs['eval/patience'] = patience

                trainer.psnr_list.append(eval_metrics['psnr'])
                psnr_running_mean = sum(trainer.psnr_list[-3:]) / len(trainer.psnr_list[-3:])

                reset_patience = False
                if eval_metrics['psnr'] > trainer.best_psnr:
                    trainer.best_psnr = eval_metrics['psnr']
                    reset_patience = True

                wandb_logs['Best PSNR'] = trainer.best_psnr

                if reset_patience:
                    patience = config['patience']

                    if epoch > 2:
                        logger.info(f"Saving best model (eval) with eval_PSNR: {trainer.best_psnr:.02f}")
                        trainer.save_checkpoints(filename=f"{config['experiment_name']}_{trainer.best_psnr:.02f}")
                else:
                    patience -= 1

                start_test_time = time.time()
                test_metrics, test_loss, _ = trainer.test()

                wandb_logs['test/time'] = time.time() - start_test_time
                wandb_logs['test/avg_loss'] = test_loss
                wandb_logs['test/avg_psnr'] = test_metrics['psnr']

            wandb_logs['epoch'] = trainer.epoch
            wandb_logs['epoch_time'] = time.time() - start_epoch_time

            logger.info(f"Eval Loss: {eval_loss:.4f} - PSNR: {eval_metrics['psnr']:.4f}, best: {trainer.best_psnr:.4f}")
            logger.info(f"Test Loss: {test_loss:.4f} - PSNR: {test_metrics['psnr']:.4f}, best: {trainer.best_psnr:.4f}")

            if config['lr_scheduler'] == 'plateau':
                trainer.lr_scheduler.step(metrics=psnr_running_mean)
            else:
                trainer.lr_scheduler.step()

            if wandb_log:
                wandb_log.on_log(wandb_logs)

            logger.info(f"Saving model...")
            trainer.save_checkpoints(filename=config['experiment_name'])
            logger.info('-' * 75)

            if patience == 0:
                logger.info(f"No update of Best PSNR value in the last {config['patience']} epochs. Stopping training.")
                sys.exit()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Training failed due to {e}")
    finally:
        logger.info("Training finished")
        sys.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_paths', type=str, nargs='+', required=True)
    parser.add_argument('--eval_dataset_name', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, help=f"Experiment name")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_dir', type=str, default='/tmp')
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--n_downsampling', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resume', type=str, default='none')
    parser.add_argument('--unet_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--apply_threshold_to', type=str, default='test', choices=['none', 'val_test', 'test', 'all'])
    parser.add_argument('--loss', type=str, nargs='+', default=['CHAR'], choices=['MSE', 'MAE', 'CHAR', 'BCE'])
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--lr_min', type=float, default=1.5e-5)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['constant', 'linear', 'cosine', 'plateau'])
    parser.add_argument('--lr_scheduler_warmup', type=int, default=10)
    parser.add_argument('--lr_scheduler_kwargs', type=eval, default={})
    parser.add_argument('--load_data_in_memory', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--overlap_test', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--patch_size_raw', type=int)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--eps', type=float, default=1.0e-08)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    args = parser.parse_args()

    args.device_id = 0 if args.device_id is None else args.device_id
    args.device = torch.device(f'cuda:{args.device_id}')
    args.device_name = torch.cuda.get_device_name(args.device_id) if torch.cuda.is_available() else 'CPU'

    logger.info("Start process ...")

    train_config = {
        'optimizer': {
            'eps': args.eps,
            'betas': [args.beta_1, args.beta_2],
            'weight_decay': args.weight_decay,
        },
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'checkpoint_dir': args.checkpoint_dir,
        'init_conv_kwargs': {
            'ratio_gin': 0,
            'ratio_gout': 0
        },
        'down_sample_conv_kwargs': {
            'ratio_gin': 0,
            'ratio_gout': 0
        },
        'resnet_conv_kwargs': {
            'ratio_gin': 0.75,
            'ratio_gout': 0.75
        },
        'train_log_every': 100,
        'train_max_value': 500}

    if args.resume != 'none':
        checkpoint_path = Path(train_config['checkpoint_dir'])
        checkpoints = sorted(checkpoint_path.glob(f"*{args.resume}*.pth"))
        assert len(checkpoints) > 0, f"Found {len(checkpoints)} checkpoints with uuid {args.resume}"
        train_config['resume'] = checkpoints[0]
        args.experiment_name = checkpoints[0].stem.rstrip('_best_psnr')
        if '_best_psnr' in checkpoints[0].stem:
            logger.info(f"Resuming from best PSNR checkpoint {checkpoints[0]}")

    if args.experiment_name is None:
        exp_name = [
            str(uuid.uuid4())[:4],
            str(args.test_dataset_name),
        ]
        args.experiment_name = '_'.join(exp_name)

    train_config['experiment_name'] = args.experiment_name
    train_config['device'] = args.device
    train_config['device_name'] = args.device_name
    train_config['use_wandb'] = args.use_wandb
    train_config['wandb_dir'] = args.wandb_dir
    train_config['wandb_project'] = args.wandb_project
    train_config['wandb_entity'] = args.wandb_entity
    train_config['unet_layers'] = args.unet_layers
    train_config['n_blocks'] = args.n_blocks
    train_config['n_downsampling'] = args.n_downsampling
    train_config['losses'] = args.loss
    train_config['lr_scheduler'] = args.lr_scheduler
    train_config['lr_scheduler_kwargs'] = args.lr_scheduler_kwargs
    train_config['lr_scheduler_warmup'] = args.lr_scheduler_warmup
    train_config['learning_rate'] = args.lr
    train_config['learning_rate_min'] = args.lr_min
    train_config['seed'] = args.seed

    args.datasets_paths = {Path(dataset).name: dataset for dataset in args.datasets_paths}

    args.train_data_path = [
        d for k, d in args.datasets_paths.items() if k not in [args.test_dataset_name, args.eval_dataset_name]]

    assert args.eval_dataset_name in args.datasets_paths.keys(), f"{args.eval_dataset_name} not in {args.datasets_paths}"
    assert args.test_dataset_name in args.datasets_paths.keys(), f"{args.test_dataset_name} not in {args.datasets_paths}"

    args.eval_data_path = [args.datasets_paths[args.eval_dataset_name]]
    args.test_data_path = [args.datasets_paths[args.test_dataset_name]]
    train_config['train_data_path'] = args.train_data_path
    train_config['eval_data_path'] = args.eval_data_path
    train_config['test_data_path'] = args.test_data_path

    train_config['train_kwargs'] = {
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': False,
        'batch_size': args.batch_size
    }

    train_config['eval_kwargs'] = {
        'shuffle': False,
        'num_workers': args.num_workers,
        'batch_size': 1
    }

    train_config['test_kwargs'] = {
        'shuffle': False,
        'num_workers': args.num_workers,
        'batch_size': 1
    }

    train_config['train_batch_size'] = train_config['train_kwargs']['batch_size']
    train_config['eval_batch_size'] = train_config['eval_kwargs']['batch_size']
    train_config['test_batch_size'] = train_config['test_kwargs']['batch_size']
    train_config['num_epochs'] = args.epochs
    train_config['patience'] = args.patience
    train_config['threshold'] = args.threshold
    train_config['load_data'] = args.load_data_in_memory == 'true'
    train_config['apply_threshold_to_train'] = True if args.apply_threshold_to == 'all' else False
    train_config['apply_threshold_to_eval'] = True if args.apply_threshold_to in ['val_test', 'all'] else False
    train_config['apply_threshold_to_test'] = True if args.apply_threshold_to in ['val_test', 'test', 'all'] else False
    train_config['test_stride'] = args.patch_size // 2 if args.overlap_test == 'true' else args.patch_size
    train_config['train_patch_size'] = args.patch_size
    train_config['train_patch_size_raw'] = args.patch_size_raw if args.patch_size_raw else args.patch_size + 128
    train_config['eval_patch_size'] = args.patch_size
    train_config['test_patch_size'] = args.patch_size

    set_seed(args.seed)

    train(train_config)
    sys.exit()


if __name__ == '__main__':
    main()
