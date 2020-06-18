import json
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import tensorflow as tf
sess_config = tf.ConfigProto()

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

def prepare_dataset():
    global coco_val
    one_shot_classes = np.array([4*i + 1 for i in range(20)])
    coco_val = siamese_utils.IndexedCocoDataset()
    coco_val.load_coco(COCO_DATA, subset="val", year="2017", return_coco=True)
    coco_val.prepare()
    coco_val.build_indices()
    coco_val.ACTIVE_CLASSES = one_shot_classes
    return coco_val