import json
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_cnn.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (128, 128)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Download plant_disease_cnn.keras from Colab and place it in the models/ folder."
    )
if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Class labels not found: {LABELS_PATH}\n"
        "Export class_names.json from Colab as described in README.md."
    )

model = tf.keras.models.load_model(MODEL_PATH)
with LABELS_PATH.open("r", encoding="utf-8") as f:
    class_names = json.load(f)

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Label count ({len(class_names)}) does not match model outputs "
        f"({model.output_shape[-1]}). Use the exact class order from training."
    )


def predict_leaf(image: Image.Image):
    if image is None:
        return {}, "Please upload a leaf image."

    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    results = {
        class_names[int(i)].replace("___", " — ").replace("_", " "):
        float(probabilities[i])
        for i in top_indices
    }
    best_index = int(top_indices[0])
    confidence = float(probabilities[best_index]) * 100
    note = (
        f"Top prediction: {class_names[best_index].replace('___', ' — ').replace('_', ' ')} "
        f"({confidence:.2f}% model confidence)."
    )
    if confidence < 70:
        note += " Confidence is limited; try a clear, close-up leaf image. This is not a confirmed diagnosis."
    else:
        note += " This is a model prediction, not a confirmed diagnosis."
    return results, note


demo = gr.Interface(
    fn=predict_leaf,
    inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top 3 predictions"),
        gr.Textbox(label="Interpretation", interactive=False),
    ],
    title="Plant Disease Detection using CNN",
    description=(
        "Upload a clear plant leaf image. The CNN returns its three most likely "
        "PlantVillage categories. Educational use only; not agricultural diagnosis."
    ),
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
