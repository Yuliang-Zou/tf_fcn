# tf_fcn

A TensorFlow Implementation of:

[CVPR 2015] Long et al. [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

**NOTE:** In this repository, we only implement the VGG16 version.


## Requirements

1. TensorFlow r0.10 (r0.11 should be fine, not sure if this can work for later versions)

2. OpenCV 2 and its Python bindings

3. Ipdb: IPython environment debugger

4. (Optional) pathos. Check the other branch for further details


## Prepare dataset

In this implementation, we use the [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/). Please do as follows to set up the dataset:

1. `mkdir data` to set up the directory of dataset

2. Download the train/val dataset and Development kit tar files, put them under the `data` folder. Unzip Development kit tar file, then unzip train/val tar file and rename the folder as `VOC2011`.

3. It should have this basic structure (under `data` directory)

```bash
$ VOCdevkit/                      # development kit
$ VOCdevkit/VOCcode               # VOC utility code
$ VOCdevkit/VOC2011               # image sets, annotations, etc.
# ... and several other directories ...
```


## Pre-trained model

```bash
mkdir model
```

We use a ImageNet pre-trained model to initialize the network, please download the npy file [here](https://drive.google.com/file/d/0B2SnTpv8L4iLRTFZb0FWenRJTlU/view?usp=sharing) and put it under the `model` folder.


## How to train

Since input images have different sizes, in order to make them as minibatch input, we used two different strategies: 1) padding to a large size (640, 640) or 2) resize to a small size (256, 256)

```bash
cd src
python train.py          # padding
python train_small.py    # resize
```

You can choose either one to run. And you can also change the `config` dictionary to use custom settings.


## Demo

```bash
cd src
python demo.py
```

You can change the `config` dictionary to use custom settings.


## Evaluation

First you should run the following code:

```bash
cd src
python test.py
```

You might want to change the used model. Check the code for futher details.

After that, you should find the following structure in `result` folder:

```bash
$ FCN8_adam_iter_10000/          # folder name depends on the model you used
$ FCN8_adam_iter_10000/gray/     # gray-scale segmentation result
$ FCN8_adam_iter_10000/rgb/      # rgb segmentation result
# ... and maybe several other directories ...
```

Then you can use the VOC2011 provided eval code to do evaluation.


## Models

- FCN32_adam_20000: [ckpt](https://drive.google.com/file/d/0B3vJudZqxciYbTRuY21WZXREV0E/view?usp=sharing), [npy](https://drive.google.com/file/d/0B2SnTpv8L4iLNEVFd2RHcUZOX00/view?usp=sharing)

- FCN16_adam_5000:  [ckpt](https://drive.google.com/file/d/0B2SnTpv8L4iLT2VuREZwUHg4cjg/view?usp=sharing)

- FCN8_adam_10000:  [ckpt](https://drive.google.com/file/d/0B2SnTpv8L4iLRExqQTVONWxTX0U/view?usp=sharing)

**Note:** When you train the shortcut version model (FCN16 and FCN8), you will need FCN32 model npy file as initialization, instead of the ImageNet pre-trained model npy file.