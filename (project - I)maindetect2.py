from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
import cv2
import flask
import numpy as np
import pywhatkit as kit
import subprocess, os, platform
from win10toast import ToastNotifier
import random
import webbrowser

toast=ToastNotifier()
toast.show_toast("Hey Yashoverman!","(Project - I) emotion analyzer (offline version) is ready",duration=15)

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

book=mood[5]

if book == 'Happy':
    webbrowser.open_new(r'file://C:\Users\Yashoverman singh\OneDrive\Documents\emotions\happy\Think Like a Monk by Jay Shetty.epub')
#     n=random.randint(0,2)
#     file_dir= r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\happy\Think Like a Monk by Jay Shetty.epub"
#     book=os.listdir(file_dir)
#     os.startfile(os.path.join(file_dir,book[n]))
elif book == 'Angry':
#     n=random.randint(0,2)
    webbrowser.open_new(r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\angry\Ikigai The Japanese Secret to a Long and Happy Life.pdf") 
#     book=os.listdir(file_dir)
#     os.startfile(os.path.join(file_dir,book[n]))
elif book == 'Surprise':
#     n=random.randint(0,5)
    webbrowser.open_new(r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\sad\Atomic-Habits-James-Clear.pdf")
#     book=os.listdir(file_dir)
#     os.startfile(os.path.join(file_dir,book[n]))
elif book == 'Neutral':


# else:
#     n=random.randint(0,5)
    webbrowser.open_new(r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\neutral\The Alchemist - Paulo Coelho.pdf")
#     book=os.listdir(file_dir)
#     os.startfile(os.path.join(file_dir,book[n]))

elif book == 'Disgust':
    webbrowser.open_new(r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\disgust\Three Thousand Stitches .pdf")

elif book == 'Fear':
    webbrowser.open_new(r"C:\Users\Yashoverman singh\OneDrive\Documents\emotions\fear\Nir_Eyal_Indistractable_How_to_Control_Your_Attention_and_Choose.pdf")