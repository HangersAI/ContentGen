import cv2
import base64
import numpy as np


def image_to_base64(image):
    img_to_byte = cv2.imencode('.jpg', image)[1].tobytes()
    byte_to_base64 = base64.b64encode(img_to_byte)
    return byte_to_base64.decode('ascii')  # chuyển về string


def base64_to_image(base64_string):
    base64_to_byte = base64.b64decode(base64_string)
    byte_to_image = np.frombuffer(base64_to_byte, np.uint8)
    image = cv2.imdecode(byte_to_image, cv2.IMREAD_COLOR)
    return image