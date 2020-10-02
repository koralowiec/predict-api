from fastapi import FastAPI
from pydantic import BaseModel, Field

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import base64

# For measuring the inference time.
import time

import os
import logging as log

debug = os.environ.get("DEBUG")
log.basicConfig(
    level=log.DEBUG if debug else log.INFO, format="%(asctime)s - %(message)s"
)

app = FastAPI()

# Print Tensorflow version
log.info("TensorFlow version: %s", tf.__version__)

# Check available GPU devices.
log.info(
    "The following GPU devices are available: %s" % tf.test.gpu_device_name()
)


# Object detection module
module_env = os.environ.get("MODULE")
log.info("MODULE env: %s", module_env)
module_path = (
    "/model_faster_rcnn" if module_env == "FASTER_RCNN" else "/model_ssd"
)

log.info("Loading module_env from: %s", module_path)

start_time = time.time()
detector = hub.load(module_path).signatures["default"]
end_time = time.time()

log.info("Loading module time: %.2f", end_time - start_time)


def run_detector(detector, img) -> dict:
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...
    ]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    log.debug(result)

    result = {key: value.numpy().tolist() for key, value in result.items()}

    log.info("Found %d objects.", len(result["detection_scores"]))
    log.info("Inference time: %.2f", end_time - start_time)
    log.debug(result)

    return result


class Base64Body(BaseModel):
    imgBase64: str = Field(..., title="Image encoded in Base64")


@app.get("/healthcheck")
def healthcheck():
    return "OK"


@app.post("/predict")
def raw_image(body: Base64Body):
    image_decoded_from_base64 = base64.b64decode(body.imgBase64)
    image = tf.image.decode_jpeg(image_decoded_from_base64, channels=3)

    return run_detector(detector, image)
