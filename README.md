# WIP

# Workshop: Human-like visual search application with small data
A workshop on how to build and deploy human-like visual search application with Siamese Mask R-CNN

Audience level: Beginner - Intermediate 

## Abstract 

_Task_: Prototype a visual search application with human-like flexibility

_Limitations_: Unaffordable price for large annotated datasets, small and limited data, computational resources

_Solution_: One-shot instance segmentation with Siamese Mask R-CNN

During the workshop we cover the building blocks of Siamese Mask R-CNN model and showcase its ability to detect and segment objects from novel categories based on a single reference image. Next, we try a single deployment. Lastly, we discuss possible limitations and improvements. At the end of the workshop, participants will have a basic understanding on how to prototype and deploy human-like visual search applications.

# Siamese Mask R-CNN

This is the official implementation of Siamese Mask R-CNN from [One-Shot Instance Segmentation](https://arxiv.org/abs/1811.11507). It is based on the [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation by [Matterport](https://github.com/matterport/Mask_RCNN).

<p align="center">
 <img src="figures/teaser_web.jpg" width=80%>
</p>

The repository includes:
- [x] Source code of Siamese Mask R-CNN
- [x] Training code for MS COCO
- [x] Evaluation on MS COCO metrics (AP)
- [x] Training and evaluation of one-shot splits of MS COCO
- [x] Training code to reproduce the results from the paper
- [x] Pre-trained weights for ImageNet
- [x] Pre-trained weights for all models from the paper
- [x] Code to evaluate all models from the paper
- [x] Code to generate result figures

## One-Shot Instance Segmentation

One-shot instance segmentation can be summed up as: Given a query image and a reference image showing an object of a novel category, we seek to detect and segment all instances of the corresponding category (in the image above ‘person’ on the left, ‘car’ on the right). Note that no ground truth annotations of reference categories are used during training.
This type of visual search task creates new challenges for computer vision algorithms, as methods from metric and few-shot learning have to be incorporated into the notoriously hard tasks ofobject identification and segmentation. 
Siamese Mask R-CNN extends Mask R-CNN - a state-of-the-art object detection and segmentation system - with a Siamese backbone and a matching procedure to perform this type of visual search.

### Prepare COCO dataset

The model requires [MS COCO](http://cocodataset.org/#home) and the [CocoAPI](https://github.com/waleedka/coco) to be added to `/data`.
```
cd data
git clone https://github.com/cocodataset/cocoapi.git
```
It is recommended to symlink the dataset root of MS COCO. 
```
ln -s $PATH_TO_COCO$/coco coco
```
If unsure follow the instructions of the [Matterport Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN#ms-coco-requirements).

### Get pretrained weights

Get the pretrained weights from the [releases menu](https://github.com/bethgelab/siamese-mask-rcnn/releases) and save them to `/checkpoints`.

## Training

To train siamese mask r-cnn on MS COCO simply follow the instructions in the [training.ipynb](training.ipynb) notebook. 
There are two model configs available, a small one which runs on a single GPU with 12GB memory and a large one which needs 4 GPUs with 12GB memory each.
The second model config is the same as used in our experiments.

To reproduce our results and train the models reported in the paper run the notebooks provided in [experiments](experiments). 
Those models need 4 GPUs with 12GB memory each.

Our models are trained on the coco 2017 training set, of which we remove the last 3000 images for validation.

## Evaluation

To evaluate and visualize a models results run the [evaluation.ipynb](evaluation.ipynb) notebook. Make sure to use the same config as used for training the model.

To evaluate the models reported in the paper run the evaluation notebook provided in [experiments](experiments). 
Each model will be evaluated 5 times to compensate for the stochastic effects introduced by randomly choosing the reference instances. 
The final result is the mean of those five runs.

We use the coco 2017 val set for testing and the last 3000 images from the training set for validation.

## Model description

Siamese Mask R-CNN is designed as a minimal variation of Mask R-CNN which can perform the visual search task described above. For more details please read the [paper](https://arxiv.org/abs/1811.11507).

<p align="center">
 <img src="figures/siamese-mask-rcnn-sketch.png" width=50%>
</p>

# Getting Started
* [demo.ipynb](samples/demo.ipynb) Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* [train_shapes.ipynb](samples/shapes/train_shapes.ipynb) shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

# Training on MS COCO
We're providing pre-trained weights for MS COCO to make it easier to start. You can
use those weights as a starting point to train your own variation on the network.
Training and evaluation code is in `samples/coco/coco.py`. You can import this
module in Jupyter notebook (see the provided notebooks for examples) or you
can run it directly from the command line as such:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last
```

You can also run the COCO evaluation code with:
```
# Run COCO evaluation on the last trained model
python3 samples/coco/coco.py evaluate --dataset=/path/to/coco/ --model=last
```

The training schedule, learning rate, and other parameters should be set in `samples/coco/coco.py`.

# Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.

In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/shapes/train_shapes.ipynb`, `samples/coco/coco.py`, `samples/balloon/balloon.py`, and `samples/nucleus/nucleus.py`.

## Requirements
miniconda and other packages listed in `environment.yml`.

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   conda 
   ```
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

