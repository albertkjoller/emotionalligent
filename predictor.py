#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:15 2020

@author: Jacobsen
"""

from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


#Evt. gøres på en anden måde med webcam eller andet
def prepare(filepath):
    IMG_SIZE = 48  # 48 in txt-based
#    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model("64x3-CNN.model")
"""
camera = cv2.VideoCapture(0)
return_value, image = camera.read()
cv2.imwrite('opencv.png', image)
del(camera)

picture = 'opencv.png'
#picture = '/Users/AlbertoK/Desktop/DTU/Januar2020/emotionalligent/thumbnail_large.jpg'

prediction = model.predict([prepare(picture)])
print(prediction)  # will be a list in a list."""

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        
        prediction = model.predict([prepare(frame)])
        print(prediction)  # will be a list in a list.
        
        text = CATEGORIES[np.where(prediction==np.max(prediction))[1][0]]
        print(text)
                
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        color = (255, 255, 255)
        thickness = cv2.FILLED

#        cv2.putText(frame, text, (20,20), font_face, scale, color, 1, cv2.LINE_AA)
        
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








"""
image = imread(picture)                                                                        
cv2.imshow(image)
#plt.axis('Off')
print(CATEGORIES[np.where(prediction==np.max(prediction))[1][0]])"""






