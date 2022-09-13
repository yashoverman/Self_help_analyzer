from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
import cv2
import numpy as np
import pywhatkit as kit
import os
from  win10toast import ToastNotifier

toast=ToastNotifier()
toast.show_toast("Hey Yashoverman!","(Project - I) emotion analyzer (online version) is ready",duration=15)

model = model_from_json(open(r"C:\Users\Yashoverman singh\OneDrive\Documents\python xd\model.json", "r").read())
model.load_weights(r"C:\Users\Yashoverman singh\OneDrive\Documents\python xd\model.h5")
face_classifier = cv2.CascadeClassifier(r"C:\Users\Yashoverman singh\Downloads\haarcascade_frontalface_default.xml")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

mood=[]

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,minNeighbors=3)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = model.predict(roi)#[0]
            label=emotion_labels[prediction.argmax()]
            mood.append(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

YouTubeRefrences=mood[5]
if YouTubeRefrences == 'Happy':
    kit.playonyt("After watching this, your brain will not be the same | Lara Boyd | TEDxVancouver")
if YouTubeRefrences == 'Surprise':
    kit.playonyt("How To READ A Book A Day To CHANGE YOUR LIFE (Read Faster Today!)| Jay Shetty")   
if YouTubeRefrences == 'Angry':
    kit.playonyt("The secret to self control | Jonathan Bricker | TEDxRainier")  
if YouTubeRefrences == 'Sad':
    kit.playonyt("Loneliest time of the year") 
if YouTubeRefrences == 'Neutral':
    kit.playonyt("The psychology of self-motivation | Scott Geller | TEDxVirginiaTech")  
if YouTubeRefrences == 'Fear':
    kit.playonyt("Rise")  
if YouTubeRefrences == 'Disgust':
    kit.playonyt("Animals")                      