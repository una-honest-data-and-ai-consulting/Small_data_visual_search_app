##TO DO: to remove spaghetti and replace with propper configs

import json
import random
import sys
import numpy as np
import skimage
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel



import io


import tensorflow as tf
sess_config = tf.ConfigProto()

COCO_DATA = 'data/coco/'
APP_TEST_DATA = 'data/coco/test2017/'
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

from model import load_model, prepare_image
from dataset import prepare_dataset

class RequestBody(BaseModel):
    number: int 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = FastAPI(
    title="VisualSearch",
    description="Proof of concept for Visual Search app powered by Siamese Mask R-CNN exclusive for Pydata Amsterdam Festival 2020 :)",
    version="1.0.0",
)
model = None

model, graph = load_model()
coco_val = prepare_dataset()

@app.get("/")
def read_root():
    msg = (
        "PyData Amsterdam Festival rulezzz!!!!!!!!"
    )
    return {"message": msg}

@app.post("/predict_by_category")
def predict_by_category(body: RequestBody):

    category = body.number
    if category not in range(1, 81):
        return {"msg": "Please indicate a number in the range from 1 to 80"}

    image_id = np.random.choice(coco_val.category_image_index[category])
    target = siamese_utils.get_one_target(category, coco_val, model.config)
    image = coco_val.load_image(image_id)

    with graph.as_default():
        prediction = model.detect([[target]], [image], verbose=0)
    
    results = prediction[0]
    json_results = json.dumps({"rois": results["rois"], "masks": results["masks"],
                               "class_ids": results["class_ids"], "scores": results["scores"]}, cls=NumpyEncoder)

    return json_results

@app.post("/predict_image")
def predict_image(image_file: UploadFile=File(...)):
    file_path = APP_TEST_DATA+image_file.filename
    image = skimage.io.imread(file_path)
    image = prepare_image(image)

    target = siamese_utils.get_one_target(np.random.choice(80), coco_val, model.config)
     
    with graph.as_default():
        prediction = model.detect([[target]], [image[0]], verbose=0)
    
    results = prediction[0]
    json_results = json.dumps({"rois": results["rois"], "masks": results["masks"],
                               "class_ids": results["class_ids"], "scores": results["scores"]}, cls=NumpyEncoder)

    return json_results