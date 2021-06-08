# DIP2021-SSNet
The context of this repository is mainly based on [DIP2021-Baseline](https://github.com/bridgeqiqi/DIP2021-FinalPJbaseline). Compared to the origin one, this repository add the model of SSNet and modify the code to offer a simple pipeline of training and testing on ShanghaiTech PartB.

# Environment & Folders

- python 3.7.
- pytorch 1.4.0
- torchvision 0.5.0
- numpy 1.20.1
- tensorboard 2.2.1
- tqdm 4.51.0

This pipeline is a simple framework for crowd counting task including four folders(*datasets*, *losses*, *models*, *optimizers*, *Make_Datasets*) and three files(*main.py*, *test.py*, *train.sh*).

- main.py: The entrance of the main program.
- test.py: Compute the MAE and RMSE metrics among testset images based on your checkpoints.
- train.sh: You can run ```sh ./train.sh```	to launch training.
- datasets: This folder contains dataloaders from different datasets.
- losses: This folder contains different customized loss functions if needed.
- models: This folder contains different models. CSRNet is provided here.
- optimizers: This folder contains different optimzers.
- Make_Datasets: This folder contains density map generation codes.

# Datasets Preparation
- ShanghaiTech PartA and PartB: [download_link](https://pan.baidu.com/s/1nuAYslz)
- UCF-QNRF: [download_link](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- NWPU: [download_link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FNWPU%2DCrowd&originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fc3ViTXA0OHd3SkRpSDBZbFQ4Mk5ZWUJtWTlMMHMtRnByckJjb2FBSmtJMXJ3P3J0aW1lPXlxTUoxbF82MkVn)
- GCC: [download link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/Eo4L82dALJFDvUdy8rBm6B0BuQk6n5akJaN1WUF1BAeKUA?e=ge2cRg)

The density map generation codes are in Make_Datasets folders.

After all density maps are generated, run ```ls -R /xx/xxx/xxx/*.jpg > train.txt```, ```ls -R /xx/xxx/xxx/*.jpg > val.txt```, ```ls -R /xx/xxx/xxx/*.jpg > test.txt``` to generate txt files for training, validating and testing.


# Quick Start for Training and Testing

- Training

run ```sh ./train.sh``` or run the following command.
```
python main.py --dataset SHHB \
--model SSNet \
--train-files .\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\train\\train.txt \
--val-files .\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\train\\val.txt \
--gpu-devices 1 \
--lr 1e-5 \
--optim adam \
--loss bceloss \
--checkpoints ./checkpoints \
--summary-writer ./runs/demo
```

- Testing

run the following command.
```
python test.py --test-files .\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\test\\test.txt --best-model .\\checkpoints\\bestvalmodel.pth
```

You may need to modify the absolute path of some files or directories in the command above or in code to conduct your training and testing.

This repository contains the pretrained parameters in `./checkpoint/mybestmodel.pth`.