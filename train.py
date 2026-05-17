import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

print("imports done")

dataset_path = "/root/.cache/kagglehub/competitions/early-detection-of-3d-printing-issues/"
images_path = dataset_path + "images/"
train = dataset_path + "train.csv"

IMG_SIZE = (640, 640)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_df = pd.read_csv(train)

# Split train into train and validation
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["has_under_extrusion"])

train_paths = [
    images_path + p
    for p in train_df["img_path"]
]

val_paths = [
    images_path + p
    for p in val_df["img_path"]
]

train_labels = train_df["has_under_extrusion"].values
val_labels = val_df["has_under_extrusion"].values

def process_image(path, label):
    # load file and decode to image tensor
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

# map preprocessing
train_ds = train_ds.map(process_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_image, num_parallel_calls=AUTOTUNE)

# shuffle train set
train_ds = train_ds.shuffle(1000)

# batch
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(BATCH_SIZE)

# prefetch for GPU efficiency
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)



print("Loading EfficientNetB0 base model...")
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))


base_model.trainable = False


print("Building model...")
model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")  # binary classification

])


print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

print("Training model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=15, batch_size=32)

loss, acc = model.evaluate(val_ds)

print("Validation Accuracy:", acc)

model.save("efficientnetb0_under_extrusion.h5")
model.save("efficientnetb0_under_extrusion_savedmodel.keras")