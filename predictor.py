#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:15 2020

@author: Jacobsen
"""


import numpy as np
import cv2
import tensorflow as tf

CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


#Evt. gøres på en anden måde med webcam eller andet
def prepare(filepath):
    IMG_SIZE = 48  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare("/Users/AlbertoK/Desktop/DTU/Januar 2020/emotionalligent/thumbnail_large.jpg")])
print(prediction)  # will be a list in a list.
print(CATEGORIES[np.where(prediction==np.max(prediction))[1][0]])






