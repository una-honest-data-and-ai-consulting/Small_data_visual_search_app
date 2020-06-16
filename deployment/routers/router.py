COCO_DATA = '../data/coco/'

import random
import numpy as np
import matplotlib.pyplot as plt
from fastapi import APIRouter, File

from lib.Mask_RCNN.samples.coco import coco
from lib import utils as siamese_utils
from ..model import prepare_model

def prepare_dataset():
    one_shot_classes = np.array([4*i + 1 for i in range(20)])
    coco_val = siamese_utils.IndexedCocoDataset()
    coco_val.load_coco(COCO_DATA, subset="val", year="2017", return_coco=True)
    coco_val.prepare()
    coco_val.build_indices()
    coco_val.ACTIVE_CLASSES = one_shot_classes
    return coco_val

router = APIRouter()

@router.post('/predict')
def model_router():
    ## load model in inference mode with checkpoints
    model = prepare_model()
    coco_val = prepare_dataset()

    category = np.random.choice(80)
    image_id = np.random.choice(coco_val.category_image_index[category])
    target = siamese_utils.get_one_target(category, coco_val, model.config)
    image = coco_val.load_image(image_id)

    results = model.detect([[target]], [image], verbose=1)
    r = results[0]

    return siamese_utils.display_results(target, image, r['rois'], r['masks'], r['class_ids'], r['scores'])