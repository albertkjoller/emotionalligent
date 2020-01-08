#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:30:21 2020

@author: Jacobsen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm



DATADIR = "/Users/Jacobsen/Desktop/cat&dogs/PetImages"

CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

"""for category in CATEGORIES:  # do emotions
    path = os.path.join(DATADIR,category)  # create path to emotions
    for img in os.listdir(path):  # iterate over each image per emotion
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        #plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
    """


IMG_SIZE = 50

#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do emotions

        path = os.path.join(DATADIR,category)  # create path to emotions
        class_num = CATEGORIES.index(category)  # get the classification  (0-6).

        for img in tqdm(os.listdir(path)):  # iterate over each image per emotion
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()


import random

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()









