# tf_fcn

A TensorFlow Implementation of:

[CVPR 2015] Long et al. [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

**NOTE:** In this repository, we only implement the VGG16 version.


## Prepare dataset

In this implementation, we use the [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/). Please do as follows to set up the dataset:

1. `mkdir data` to set up the directory of dataset

2. Download the train/val dataset and Development kit tar files, put them under the `data` folder. Unzip Development kit tar file, then unzip train/val tar file and rename the folder as `VOC2011`.

3. It should have theis basic structure (under `data` directory)

```bash
$ VOCdevkit/                      # development kit
$ VOCdevkit/VOCcode               # VOC utility code
$ VOCdevkit/VOC2011               # image sets, annotations, etc.
# ... and several other directories ...
```


## Pre-trained model

We use a ImageNet pre-trained model to initialize the network, please download the npy file [here](https://drive.google.com/file/d/0B2SnTpv8L4iLRTFZb0FWenRJTlU/view?usp=sharing)


## How to train

```bash
cd src
python train.py
```

You can change the `config` dictionary to use custom settings.


## Demo

```base
cd src
python demo.py
```

You can change the `config` dictionary to use custom settings.


## Evaluation

Not yet finished...