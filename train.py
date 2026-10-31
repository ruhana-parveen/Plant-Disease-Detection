"""Train the PlantVillage CNN and export model + exact class label order."""
import json
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Explicit, non-overlapping 80/20 dataset split.
(train_raw, val_raw), info = tfds.load(
    "plant_village",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
)

num_classes = info.features["label"].num_classes
class_names = info.features["label"].names

def preprocess(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = (
    train_raw
    .shuffle(2000, seed=42)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    val_raw
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(OUTPUT_DIR / "plant_disease_cnn.keras")
with (OUTPUT_DIR / "class_names.json").open("w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print(f"Saved model and {len(class_names)} class labels to {OUTPUT_DIR}/")
print("Final validation accuracy:", history.history["val_accuracy"][-1])
