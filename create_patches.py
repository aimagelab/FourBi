import argparse
from data.process_image import PatchImage


def main():
    parser = argparse.ArgumentParser(description='create patches')
    parser.add_argument('--path_dst', type=str, help=f"Destination folder path")
    parser.add_argument('--path_src', type=str, help="The path witch contains the images")
    parser.add_argument('--patch_size', type=int, help="Patch size", default=384)
    parser.add_argument('--overlap_size', type=int, help='Overlap size', default=192)
    args = parser.parse_args()

    patcher = PatchImage(patch_size=args.patch_size, overlap_size=args.overlap_size, destination_root=args.path_dst)
    patcher.create_patches(root_original=args.path_src)


if __name__ == '__main__':
    main()
