# FourBi

This repository contains the official implementation for our paper [Binarizing Documents by Leveraging both Space and Frequency](https://arxiv.org/abs/2404.17243).
If you find it useful, please cite it as:
```
@inproceedings{pippi2023handwritten,
  title={{Binarizing Documents by Leveraging both Space and Frequency}},
  author={Quattrini, Fabio and Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2023},
  organization={Springer}
}
```

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

## Models
We release the pre-trained weights for the FourBi variants trained on DIBCO benchmarks. 

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow">Testing data</th>
    <th class="tg-c3ow">URL</th>
    <th class="tg-baqh">PSNR</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">0</td>
    <td class="tg-c3ow">H-DIBCO 2010</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/9e1a_HDIBCO10.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">23.37</td>
  </tr>
  <tr>
    <td class="tg-c3ow">1</td>
    <td class="tg-c3ow">DIBCO 2011</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/b9cd_DIBCO11.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">22.26</td>
  </tr>
  <tr>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">H-DIBCO 2012</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/0f90_HDIBCO12.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">24.29</td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">DIBCO 2013</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/ed5a_DIBCO13.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">24.17</td>
  </tr>
  <tr>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">H-DIBCO 2014</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/2bd8_HDIBCO14.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">25.18</td>
  </tr>
  <tr>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">H-DIBCO 2016</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/c004_HDIBCO16.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">19.74</td>
  </tr>
  <tr>
    <td class="tg-c3ow">6</td>
    <td class="tg-c3ow">DIBCO 2017</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/b2d1_DIBCO17.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">19.66</td>
  </tr>
  <tr>
    <td class="tg-c3ow">7</td>
    <td class="tg-c3ow">H-DIBCO 2018</td>
    <td class="tg-c3ow"><a href="https://github.com/aimagelab/FourBi_7/releases/download/Checkpoints/3d22_HDIBCO18.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    <td class="tg-amwm">20.92</td>
  </tr>


</tbody>
</table>


 
