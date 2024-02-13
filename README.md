# FourBi_7

## Setup
To run this project, we used `python 3.11.7` and `pytorch 2.2.0` 
```bash
conda create -n fourbi python=3.11.7
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install opencv-python wandb pytorch-ignite
```

## Inference
To run the model on a folder with images, run with the following command
```
python binarize.py <path to checkpoint> --src <path to the test images folder> 
--dst <path to the output folder>
```

## Training
The model is trained on patches, then evaluated and tested on complete documents. We provide the code to create the patches and train the model.
For example, to train on H-DIBCO12, first download the dataset from http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/. Create a folder, then place the images in a sub-folder named "imgs" and the ground truth in a sub-folder named "gt_imgs". Then run the following command:
```
python create_patches.py --path_src <path to the dataset folder> 
--path_dst <path to the folder where the patches will be saved> 
--patch_size <size of the patches> --overlap_size <size of the overlap>
```
To launch the training, run the following command:
```
python train.py --datasets_paths <all datasets paths> 
--eval_dataset_name <name of the validation dataset> 
--test_dataset_name <name of the validation dataset>
```
 
