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


def prepare(filepath):
    IMG_SIZE = 48  # 48 in txt-based
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("/Users/AlbertoK/Desktop/DTU/Januar2020/emotionalligent/dropoutCNN.model")


key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        key = cv2.waitKey(1)
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prediction = model.predict([prepare(grayFrame)])
        print(prediction)  # will be a list in a list.
        
        text = CATEGORIES[np.where(prediction==np.max(prediction))[1][0]]
        print(text)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (0,0,0)
        thickness = cv2.FILLED
        
        cv2.putText(grayFrame, text, (200, 200), font, 1, color, thickness=2)
        cv2.imshow("Capturing", grayFrame)
        
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break



