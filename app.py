##TO DO: to remove spaghetti and replace with propper configs

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

from model import prepare_model

class RequestBody(BaseModel):
    number: int

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = FastAPI()

def prepare_dataset():
    one_shot_classes = np.array([4*i + 1 for i in range(20)])
    coco_val = siamese_utils.IndexedCocoDataset()
    coco_val.load_coco(COCO_DATA, subset="val", year="2017", return_coco=True)
    coco_val.prepare()
    coco_val.build_indices()
    coco_val.ACTIVE_CLASSES = one_shot_classes
    return coco_val

def load_model():
    global model
    model = prepare_model()
    global graph
    graph = tf.get_default_graph()
    return model, graph

coco_val = prepare_dataset()
print("dataset prepared to go")
model, graph = load_model()
print("model and graph loaded")

@app.get("/")
def read_root():
    msg = (
        "PyData Amsterdam Festival rulezzz!!!!!!!!"
    )
    return {"message": msg}

@app.post("/model_test")
def model_test(body: RequestBody):

    category = body.number
    image_id = np.random.choice(coco_val.category_image_index[category])
    target = siamese_utils.get_one_target(category, coco_val, model.config)
    image = coco_val.load_image(image_id)

    with graph.as_default():
        prediction = model.detect([[target]], [image], verbose=1)
    
    results = prediction[0]
    
    json_results = json.dumps({"rois": results["rois"], "masks": results["masks"],
                               "class_ids": results["class_ids"], "scores": results["scores"]}, cls=NumpyEncoder)

    return json_results
