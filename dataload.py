#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:30:21 2020

@author: Jacobsen
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2



CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

filename = '/Users/AlbertoK/Desktop/fer2013.csv'
data = pd.read_csv(filename, sep=",")
data = data.drop('Usage',axis=1)

X = []

emotion = np.array(data.emotion)
pos=[]  
    
for i in range(len(CATEGORIES)):
    if i == 1:
        i = i+1
    else:
        number = np.where(emotion==i)[0]
        pos.extend(list(number[:4000]))

pos = np.array(np.sort(pos))
y = []

pixels = data.pixels.to_numpy()

#Every picture
for i in range(len(pos-1)):
    img_array = list(map(int, list(pixels[pos[i]].split(" "))))
    IMG_SIZE = int(np.sqrt(np.size(img_array)))
    X.append(np.reshape(img_array,(IMG_SIZE, IMG_SIZE)))
    
    if emotion[pos[i]] >= 2:
        y.append(emotion[pos[i]]-1)
    else:
        y.append(emotion[pos[i]])
    
X = list(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()







