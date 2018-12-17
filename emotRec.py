import cv2
import glob
import random
import numpy as np
import time




video_capture = cv2.VideoCapture(0) 

filenumber = 0
fileName = ""

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "disgust", "happy", "surprise"] #Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  #Initialize fisher face classifier FisherFaceRecognizer_create()
fishface.read("trained_net_5_emotions.yml") # Load trained fisher face classifier 

while True:
    frame = video_capture.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", frame)
    
    try:
        openImage = cv2.imread(fileName)
        gr = cv2.cvtColor(openImage, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Result", openImage)
    except:
        print("Can't open image")
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        cv2.destroyWindow("Result")

    if cv2.waitKey(2) & 0xFF == ord('p'):
        print("PHOTO WAS TAKEN")
        
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            print("No face!")
            continue
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file:")
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            fileName = "frames\\%s.png" %(filenumber)
            cv2.imwrite(fileName, out) #Write image

        filenumber += 1 #Increment image number
        file = glob.glob(fileName)
        for i in file:
            image = cv2.imread(i)
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data_for_prediction = []
        data_for_prediction.append(gray2)

        for img in data_for_prediction:
            net_predicted, confidence = fishface.predict(img)
            print("You have a/an ", emotions[net_predicted], " face")
            cv2.putText(img, emotions[net_predicted], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 0, 125))
            out = cv2.resize(gray2, (350, 350))
            cv2.imwrite(fileName, out)    
        
    if cv2.waitKey(3) & 0xFF == ord('q'): #close the camera
        break	

video_capture.release()
cv2.destroyAllWindows()

