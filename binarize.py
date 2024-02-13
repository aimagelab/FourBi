import argparse
import torch
from pathlib import Path
from trainer.fourbi_trainer import FourbiTrainingModule
from data.test_dataset import FolderDataset
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize a folder of images')
    parser.add_argument('model', type=str, metavar='PATH', help='path to the model file')
    parser.add_argument('--src', type=str, required=True, help='path to the folder of input images')
    parser.add_argument('--dst', type=str, required=True, help='path to the folder of output images')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size when processing patches')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    fourbi = FourbiTrainingModule(config={'resume': args.model}, device=device, make_loaders=False)

    dataset = FolderDataset(Path(args.src), patch_size=args.patch_size, overlap=True, transform=transforms.ToTensor())
    fourbi.test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    fourbi.config['test_patch_size'] = args.patch_size
    fourbi.config['test_stride'] = args.patch_size // 2
    fourbi.config['eval_batch_size'] = args.batch_size

    for i, sample in enumerate(fourbi.folder_test()):
        key = list(sample.keys())[0]
        img, pred, gt = sample[key]
        src_img_path = Path(key)

        dst_img_path = dst / (src_img_path.stem + '.png')
        pred.save(str(dst_img_path))
        print(f'({i + 1}/{len(dataset)}) Saving {dst_img_path}')

    print('Done.')

