# WIP

# Workshop: Human-like visual search application with small data
A workshop on how to prototype and deploy a visual search application based on Siamese Mask R-CNN

Audience level: Beginner - Intermediate 

## Workshop description

_Task_: Prototype a visual search application with human-like flexibility

_Limitations_: Unaffordable price for large annotated datasets, small data.

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

4. Get the pretrained weights from the [releases menu](https://github.com/EzheZhezhe/Small_data_visual_search_app/releases) and save them to `/checkpoints`.
 
### Prepare COCO dataset

# TODO: finish this part
The model requires [MS COCO 2017 Train and Val images, Train/Val annotations](http://cocodataset.org/download) and the [CocoAPI](https://github.com/waleedka/coco) to be added to `/data`.
```
cd data
git clone https://github.com/cocodataset/cocoapi.git
```

## Whom I talk to?
Alyona Galyeva - <alyona.galyeva@gmail.com>

## Credits

[Siamese Mask R-CNN](https://github.com/bethgelab/siamese-mask-rcnn)

[Mask R-CNN](https://github.com/matterport/Mask_RCNN)