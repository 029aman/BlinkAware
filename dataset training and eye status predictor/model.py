from h5py._hl import dataset
from scipy.sparse.construct import random
import tensorflow as tf
import cv2 as cv 
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
import pathlib

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPooling2D


dataset_url = "file:///C:/Users/029am/Downloads/eye.tgz"
dataset_path = tf.keras.utils.get_file('eye', origin=dataset_url, cache_dir='.', untar=True)

dataset_path = pathlib.Path(dataset_path)

open = list(dataset_path.glob('open/*'))
closed = list(dataset_path.glob('closed/*'))

eye_status = {
    'open' : open,
    'closed' : closed
}

eye_status_label = {
    'open' : 1,
    'closed' : 0
}


x, y = [] , []
for eye_folder, eye_status_img in eye_status.items():
    for image in eye_status_img:
        img = cv.imread(str(image))
        img_resized = cv.resize(img, (100, 100))
        x.append(img_resized)
        y.append(eye_status_label[eye_folder])


x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

model = Sequential([
    layers.Conv2D(32, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 4, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_scaled, y_train, epochs=10)

model.save('models/eyestatus.h5', overwrite=True)


loaded_model = load_model('models/eyestatus.h5')

loaded_model.evaluate(x_test_scaled, y_test)
