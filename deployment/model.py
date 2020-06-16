import random
import numpy as np

from collections import OrderedDict


import tensorflow as tf
import sys
sess_config = tf.ConfigProto()

MODEL_DIR = '../logs/'
  
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config
   



class SmallEvalConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'coco'
    EXPERIMENT = 'evaluation'
    CHECKPOINT_DIR = 'checkpoints/'
    NUM_TARGETS = 1

config = SmallEvalConfig()

# Provide training schedule of the model
# When evaluationg intermediate steps the tranining schedule must be provided
train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}

def prepare_model():
    model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_checkpoint('../checkpoints/small_siamese_mrcnn_0160.h5', training_schedule=train_schedule)
    return model
