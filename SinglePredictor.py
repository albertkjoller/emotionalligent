#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:29:37 2020

@author: AlbertoK
"""

from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def prepare(filepath):
    IMG_SIZE = 48  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

picture = '/Users/AlbertoK/Desktop/DTU/Januar2020/emotionalligent/thumbnail_large.jpg'

prediction = model.predict([prepare(picture)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[np.where(prediction==np.max(prediction))[1][0]])


image = imread(picture)
plt.imshow(image)
plt.axis('Off')
plt.show()
