# WIP

# Workshop: Human-like visual search application with small data
A workshop on how to prototype and deploy a visual search application based on Siamese Mask R-CNN

Audience level: Beginner - Intermediate 

## Workshop description

_Task_: Prototype a visual search application with human-like flexibility

_Limitations_: Unaffordable price for large annotated datasets, small data, limited computational resources

_Solution_: One-shot instance segmentation with Siamese Mask R-CNN

During the workshop we will learn more about one-shot instance segmentation and cover the building blocks of Siamese Mask R-CNN model. Next, we will try a single deployment. Lastly, we will discuss possible limitations and improvements. At the end of the workshop, participants will have a basic understanding on how to prototype and deploy human-like visual search applications.

## One-Shot Instance Segmentation

One-Shot Instance Segmentation can be summed up as: Given a query image and a reference image showing an object of a novel category, detect and segment all instances of the corresponding category. Note, that no ground truth annotations of reference categories are used during training.

## Model description

Siamese Mask R-CNN extends Mask R-CNN (a state-of-the-art object detection and segmentation system) with a Siamese backbone and a matching procedure to perform this type of visual search. For more details please read the [original paper](https://arxiv.org/abs/1811.11507).

<p align="center">
 <img src="assets/siamese-mask-rcnn.png" width=50%>
</p>

# Getting Started

## Requirements
- [miniconda](https://docs.conda.io/en/latest/miniconda.html) and other packages listed in `environment.yml`.
- Prepared [MS COCO Dataset](http://cocodataset.org/#download)

## Installation

1. Clone this repository

2. Install dependencies
   ```bash
   conda env create -f environment.yml
   ```

3. [Prepare COCO dataset](#Prepare-COCO-dataset)   

4. Get the pretrained weights from the [releases menu](...) and save them to `/checkpoints`.
 

### Prepare COCO dataset

The model requires [MS COCO 2017 Train and Val images, Train/Val annotations](http://cocodataset.org/download) and the [CocoAPI](https://github.com/waleedka/coco) to be added to `/data`.
```
cd data
git clone https://github.com/cocodataset/cocoapi.git
```
It is recommended to symlink the dataset root of MS COCO. 
```
ln -s $PATH_TO_COCO$/coco coco
```
If unsure follow the instructions of the [Matterport Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN#ms-coco-requirements).

* [demo.ipynb](samples/demo.ipynb) Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* [train_shapes.ipynb](samples/shapes/train_shapes.ipynb) shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 


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

