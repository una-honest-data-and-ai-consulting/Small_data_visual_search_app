"""
Small data visual search app 
App API Example

Copyright (c) 2020 Alyona Galyeva
Licensed under the MIT License (see LICENSE for details)
------------------------------------------------------------
"""
from starlette.config import Config

config = Config(".env")

APP_TEST_DATA: str = config("APP_TEST_DATA", cast=str)
COCO_DATA: str = config("COCO_DATA", cast=str)
MASK_RCNN_MODEL_PATH: str = config("MASK_RCNN_MODEL_PATH", cast=str)
MODEL_DIR: str = config("MODEL_DIR", cast=str)