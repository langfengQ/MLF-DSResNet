# Multi-Level Firing with Spiking DS-ResNet: Enabling Better and Deeper Directly-Trained Spiking Neural Networks
## IJCAI 2022
This repository contains Python implementation of MLF & spiking DS-ResNet.

## 1. Publication
If you find this repository useful, please consider citing the following paper:

<pre>
@inproceedings{LangIJCAI2022,
    title = {Multi-Level Firing with Spiking DS-ResNet: Enabling Better and Deeper Directly-Trained Spiking Neural Networks},
    author = {Lang Feng and Qianhui Liu and Huajin Tang and De Ma and Gang Pan},
    booktitle = {IJCAI},
    year = {2022},
}
</pre>


## 2. Datasets
* Download link of DVS-gesture: https://research.ibm.com/interactive/dvsgesture/.
Put file `DvsGesture.tar.gz` in the path `./data/DVS_Gesture/source_DvsGesture/`, then unzip `DvsGesture.tar.gz`.

* Download link of CIFAR10-DVS: https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2.
Put file `airplane.zip`～`horse.zip` in the path `./data/DVS_CIFAR10/source_DvsCIFAR10/`, then unzip `airplane.zip`～`horse.zip`.

* CIFAR10 dataset can be downloaded online.

## 3. Dependencies:
* python 3.7.10
* numpy 1.19.5
* torch 1.9.0+cu111
* torchvision 0.10.0+cu111
* tensorboardX 2.4
* h5py 3.3.0

## 4. Preprocessing
DVS-gesture and CIFAR10-DVS need to be pre-processed. The syntax is as follow,
```
python DVS_CIFAR10_preprocess.py
python DVS_Gesture_preprocess.py
```

## 5. Traning
To train a new model, the basic syntax is like:
```
python train_for_cifar10.py
python train_for_gesture.py
python train_for_dvscifar10.py
```

