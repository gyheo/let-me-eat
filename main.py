import numpy as np
import flask
import io

from keras.applications import ResNet50
from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils
from PIL import Image


app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Image Resize
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"Success": False}

    if flask.request.method == "POST":
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image, target=(224, 224))

        model_predict = model.predict(image)
        results = imagenet_utils.decode_predictions(model_predict)

        data["predictions"] = []

        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        data["Success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=80)
