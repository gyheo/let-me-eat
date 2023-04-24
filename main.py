import openai
import numpy as np
import base64
import io
import os

from flask import Flask, render_template, request
from keras.applications import ResNet50
from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils

from PIL import Image
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = None


def load_model():
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Image resize and preprocess
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/foods", methods=["POST"])
def get_foods():
    data = {"Success": False}

    if request.method == "POST":
        image = request.files["image"].read()

        # base64 encoding
        image_base64 = "data:{};base64,{}".format(
            request.files["image"].content_type, base64.b64encode(image).decode())
        image = Image.open(io.BytesIO(image))

        image = prepare_image(image, target=(224, 224))

        model_predict = model.predict(image)
        results = imagenet_utils.decode_predictions(model_predict)

        data["predictions"] = []

        for (imagenetID, label, prob) in results[0]:
            item = {"label": label, "probability": float(prob)}
            data["predictions"].append(item)

        data["Success"] = True

    return render_template("food.html", data=data, image_base64=image_base64)


@app.route("/answer", methods=["POST"])
def get_questions():
    label = request.form.get("label")
    question = request.form.get("question")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "{} {}".format(label, question)}
        ]
    )

    content = completion.choices[0].message.content
    title = "{} {}".format(label, question)
    return render_template("answer.html", title=title, content=content)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=80)
