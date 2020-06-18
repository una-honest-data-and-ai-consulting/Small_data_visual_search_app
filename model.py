import random
import numpy as np
import tensorflow as tf
import sys


MODEL_DIR = 'logs/'
COCO_DATA = 'data/coco/'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
    
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

def load_model():
    global model
    model = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_checkpoint('checkpoints/small_siamese_mrcnn_0160.h5')
    global graph
    graph = tf.get_default_graph()
    return model, graph


def prepare_image(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, max_dim=config.IMAGE_MAX_DIM, min_scale=config.IMAGE_MIN_SCALE, mode=config.IMAGE_RESIZE_MODE)
    return image
