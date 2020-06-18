##TO DO: to remove spaghetti and replace with propper configs

import json
import random
import sys
import numpy as np
import skimage
import matplotlib.pyplot as plt
from fastapi import FastAPI, File
from pydantic import BaseModel
from starlette.requests import Request

from PIL import Image
import io
from keras_preprocessing.image import img_to_array

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

from model import load_model, prepare_image
from dataset import prepare_dataset

class RequestBody(BaseModel):
    number: int

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = FastAPI()
model = None

coco_val = prepare_dataset()
model, graph = load_model()

@app.get("/")
def read_root():
    msg = (
        "PyData Amsterdam Festival rulezzz!!!!!!!!"
    )
    return {"message": msg}

@app.post("/predict")
def predict(body: RequestBody):

    category = body.number
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
def predict_image(image_file: bytes=File(...)):
    image = Image.open(io.BytesIO(image_file))
    
    image = img_to_array(image)

    return {"msg": image}
    #category = np.random.choice(80)
    #image_id = np.random.choice(coco_val.category_image_index[category])
    #target = siamese_utils.get_one_target(category, coco_val, model.config)
    #image = coco_val.load_image(image_id)
 
     
    #with graph.as_default():
        #prediction = model.detect([[target]], [image], verbose=0)
    
    #results = prediction[0]
    
    #json_results = json.dumps({"rois": results["rois"], "masks": results["masks"],
                               #"class_ids": results["class_ids"], "scores": results["scores"]}, cls=NumpyEncoder)

    #return json_results