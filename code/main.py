from fastapi import FastAPI, Depends
import tensorflow as tf
import tensorflow_hub as hub
import os
import logging as log
import time

from object_detector_from_base64 import ObjectDetectorFromBase64

debug = os.environ.get("DEBUG")
log.basicConfig(
    level=log.DEBUG if debug else log.INFO, format="%(asctime)s - %(message)s"
)

app = FastAPI()

# Print Tensorflow version
log.info("TensorFlow version: %s", tf.__version__)

# Check available GPU devices.
log.info("The following GPU devices are available: %s" % tf.test.gpu_device_name())


# Object detection module
module_env = os.environ.get("MODULE")
log.info("MODULE env: %s", module_env)
module_path = "/model_faster_rcnn" if module_env == "FASTER_RCNN" else "/model_ssd"

log.info("Loading module_env from: %s", module_path)

start_time = time.time()
tf_hub_module = hub.load(module_path).signatures["default"]
end_time = time.time()

log.info("Loading module time: %.2f", end_time - start_time)

object_detector = ObjectDetectorFromBase64(tf_hub_module)


@app.get("/healthcheck")
def healthcheck():
    return "OK"


@app.post("/predict")
def detect_objects_base64(results: dict = Depends(object_detector)):
    return results
