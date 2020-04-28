from flask import Flask, logging, request, render_template
import os
import sys
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO
import skimage.draw
import numpy as np

from keras.models import load_model

app = Flask(__name__)
log = logging.create_logger(app)
log.setLevel("INFO")

ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_PATH = "models/mask_rcnn_nail_0030.h5"

global _model
global _graph


def prepare_model():
    class NailConfig(Config):  # extension of mrcnn.config
        NAME = "nail"
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1  # Background + nail
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = 0.9
        GPU_COUNT = 1

    config = NailConfig()
    # config.display()

    global _model
    model_folder_path = os.path.abspath("./models/") + "/"
    _model = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=model_folder_path)

    _model.load_weights(MODEL_PATH, by_name=True)

    # model0 = load_model(MODEL_PATH)
    # model0.summary()

    global _graph
    _graph = tf.compat.v1.get_default_graph()


prepare_model()


def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img


def encode(image) -> str:
    # convert image to bytes
    with BytesIO() as output_bytes:
        PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
        # Note JPG is not a vaild type here
        PIL_image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


###################################################################################

@app.route("/upload", methods=["POST"])
def predict():
    reqdata = request.get_json()
    base64_string = reqdata['buffer']
    image = decode(base64_string)

    global _model
    global _graph
    with _graph.as_default():
        r = _model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])

    respImage = encode(splash)

    return respImage, 201


############################################################################

@app.route("/", methods=["GET"])
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
