#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:15 2020

@author: Jacobsen
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time


CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def prepare(filepath):
    IMG_SIZE = 48  # 48 in txt-based
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("/Users/AlbertoK/Desktop/kode/128-256-128-64-32-6-dropout50-50epochs.model")




key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
#video = cv2.VideoCapture('/Users/AlbertoK/Desktop/done/færdig god animation.mp4')

emoGraph = []
angry = []
fear = []
happy = []
sad = []
surprise = []
neutral = []

cascPath = '/Users/AlbertoK/Desktop/kode/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


start = time.time()
x, y, w, h = 0, 0, 48, 48

while True:
    try:
        check, frame = webcam.read()
        key = cv2.waitKey(1)
        #ret, videoframe = video.read()
        
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayFrame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw a rectangle around the faces
        for x, y, w, h in faces:
            rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 50, 255), 2)
    
        ROI = grayFrame[y:y+h, x:x+w]
            
        ROI = ROI.astype('float32') 
        
        prediction = model.predict([prepare(ROI)])
        
        
        pos = np.where(prediction==np.max(prediction))[1][0]
        text = CATEGORIES[pos]
        emoGraph.append(pos)
                
        angry.append(prediction[0][0])
        fear.append(prediction[0][1])
        happy.append(prediction[0][2])
        sad.append(prediction[0][3])
        surprise.append(prediction[0][4])
        neutral.append(prediction[0][5])
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (90, 50, 255)
        thickness = cv2.FILLED
        
        cv2.putText(frame, text, (200, 200), font, 1, color, thickness=2)
        
        cv2.imshow("Emotionalligent", (0,0,0))
        

        
        
        if key == ord('q') & 0xff:
            print("Turning off camera.")
            webcam.release()
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)
            print("Camera off.")
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        cv2.destroyAllWindows()
        for i in range (1,5):
            cv2.waitKey(1)
        print("Camera off.")
        print("Program ended.")

        break
    
    
end = time.time()

#happy, surprise, neutral, fear, angry, sad

graph = []

for i in range(len(emoGraph)):
    if emoGraph[i] == 0:
        graph.append(-2)
    if emoGraph[i] == 1:
        graph.append(-1)
    if emoGraph[i] == 2:
        graph.append(2)
    if emoGraph[i] == 3:
        graph.append(-3)
    if emoGraph[i] == 4:
        graph.append(1)
    if emoGraph[i] == 5:
        graph.append(0)



time = (end - start)

frame_count = np.size(graph)
time = np.arange(0, time, time/frame_count)


plt.plot(time, graph)

yaxis = [-3,-2,-1,0,1,2]
emotions = ['sad', 'angry', 'fear', 'neutral', 'surprise', 'happy']
plt.yticks(yaxis,emotions)

plt.ylabel("Emotions")
plt.xlabel("Time")
plt.show()


fig, axs = plt.subplots(6, sharex=True, sharey=True)
fig.suptitle('Emotions')
axs[0].plot(time, angry)
axs[0].set_title('Angry')

axs[1].plot(time, fear)
axs[1].set_title('Fear')

axs[2].plot(time, happy)
axs[2].set_title('Happy')

axs[3].plot(time, sad)
axs[3].set_title('Sad')

axs[4].plot(time, surprise)
axs[4].set_title('Surprise')

axs[5].plot(time, neutral)
axs[5].set_title('Neutral')

for ax in axs.flat:
    ax.set(xlabel='Time')
    
for ax in axs.flat:
    ax.label_outer()
    
plt.subplots_adjust(hspace=2)
plt.ylabel('Percentage')
plt.show()







