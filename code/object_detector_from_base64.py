import tensorflow as tf
from pydantic import BaseModel, Field
import logging as log
import base64
import time


class Base64Body(BaseModel):
    imgBase64: str = Field(
        ...,
        title="Image encoded in Base64",
        example="/9j/4Az7oQTg/IjIrVkcV ... ZhleJ/hPXg1ag70wwB4rq6pKHwaNa3KGRt6knopwK6urqkD/2Q==",
    )


class ObjectDetectorFromBase64:
    def __init__(self, tf_hub_module):
        self.tf_hub_module = tf_hub_module

    def __call__(self, req_body: Base64Body) -> dict:
        image_decoded_from_base64 = base64.b64decode(req_body.imgBase64)
        image = tf.image.decode_jpeg(image_decoded_from_base64, channels=3)
        converted_img = tf.image.convert_image_dtype(image, tf.float32)[
            tf.newaxis, ...
        ]

        start_time = time.time()
        result = self.tf_hub_module(converted_img)
        end_time = time.time()

        result = {key: value.numpy().tolist() for key, value in result.items()}

        log.info("Found %d objects.", len(result["detection_scores"]))
        log.info("Inference time: %.2f", end_time - start_time)
        log.debug(result)

        return result
