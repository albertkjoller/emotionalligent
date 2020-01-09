#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:30:21 2020

@author: Jacobsen
"""

import numpy as np
import pandas as pd



CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

filename = '/Users/AlbertoK/Desktop/fer2013.csv'
data = pd.read_csv(filename, sep=",")
data = data.drop('Usage',axis=1)

X = []

disgust = np.where(data.emotion == 1)[0]
data = data.drop(disgust, axis=0)

y = np.array(data.emotion)

pixels = data.pixels.to_numpy()
for i in range(len(pixels)):
    img_array = list(map(int, list(pixels[i].split(" "))))
    IMG_SIZE = int(np.sqrt(np.size(img_array)))
    X.append(np.reshape(img_array,(IMG_SIZE, IMG_SIZE)))
    
    if y[i] >= 2:
        y[i] = y[i]-1      
    

X = list(X)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()







