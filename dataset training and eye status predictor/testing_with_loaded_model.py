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


loaded_model = load_model('models/eyestatus.h5')

x = []
img = cv.imread('testingSamples/6.png')
img2 = cv.imread('testingSamples/2.png')


img = cv.resize(img , (100, 100))
img2 = cv.resize(img2 , (100, 100))


x.append(img)
x.append(img2)


x = np.array(x)
# cv.imshow('frame', img)
x_scaled=x/255


print(len(x))
predictions = loaded_model.predict(x)


score = tf.nn.softmax(predictions[0])
print(np.argmax(score))

