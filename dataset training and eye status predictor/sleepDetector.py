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
import time as tm
from pygame import time, mixer

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPooling2D

face=cv.CascadeClassifier('cascadeClassifier/haarcascade_face.xml')
eye = cv.CascadeClassifier('cascadeClassifier/haarcascade_eye.xml')

model =load_model('models/eyestatus.h5')

video = cv.VideoCapture(0)

score=0
initial_time=tm.time()

mixer.init()
mixer.music.load('audio/alert.ogg')


if(video.isOpened()==False):
    print("video path not satisfied")
else:
    while(video.isOpened()==True):
        flag, frame= video.read()
        cv.imshow('frame', frame)

        if(flag==True):
            faces = face.detectMultiScale(frame)
            for fx, fy, fw, fh in faces:
                cropped_face = frame[fy:fy+fh, fx:fx+fw]
                eyes = eye.detectMultiScale(cropped_face)
                cropped_face_display = cv.resize(cropped_face, (200, 200))
                # cv.imshow('face', cropped_face_display)

                x = []
                for ex, ey, ew, eh in eyes:
                    cropped_eye = cropped_face[ey:ey+eh , ex:ex+ew]
                    cropped_eye = cv.resize(cropped_eye, (100, 100))
                    # cv.imshow('eye', cropped_eye)
                    x.append(cropped_eye)
                x = np.array(x)
                x_scaled = x/255 
                # print(len(x))
                if(len(x)==2):
                    # cv.imshow('eye', cropped_eye)
                    predictions = model.predict(x)
                    probablities = tf.nn.softmax(predictions[0])
                    if(np.argmax(probablities) == 1):
                        # print("open")
                        score=score+1
                    else:
                        # print("closed")
                        score=score-1
            differnce_time= (tm.time()-initial_time)
            if differnce_time > 2:
                if(score >= 0):
                    print("open")
                    initial_time=tm.time()
                else:
                    print("closed")
                    mixer.music.play()
                    while mixer.music.get_busy():
                        time.Clock().tick(2)
                    initial_time=tm.time()    
                score=0    

            k=cv.waitKey(1) & 0xFF 
            if k == 27:
                break
cv.destroyAllWindows()
