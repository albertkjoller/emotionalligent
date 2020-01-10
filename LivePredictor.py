#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:15 2020

@author: Jacobsen
"""

import numpy as np
import cv2
import sys
import tensorflow as tf

CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def prepare(filepath):
    IMG_SIZE = 48  # 48 in txt-based
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("/Users/AlbertoK/Desktop/kode/20epochs.model")


key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
emoGraph = []

cascPath = '/Applications/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    try:
        check, frame = webcam.read()
        #print(frame) #prints matrix values of each framecd 
        key = cv2.waitKey(1)
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayFrame = np.array(grayFrame, dtype='uint8')
        
        faces = faceCascade.detectMultiScale(grayFrame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 50, 255), 2)
        
        ROI = grayFrame[y:y+h, x:x+w]
        
        prediction = model.predict([prepare(ROI)])
        
        
        pos = np.where(prediction==np.max(prediction))[1][0]
        text = CATEGORIES[pos]
        emoGraph.append(pos)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (90, 50, 255)
        thickness = cv2.FILLED
        
        cv2.putText(frame, text, (200, 200), font, 1, color, thickness=2)
        
        cv2.imshow("Emotionalligent", frame)
        
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            print(emoGraph)
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break



