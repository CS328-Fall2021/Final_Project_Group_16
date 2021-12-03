import cv2 as cv
import numpy as np
import dlib, os, imutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

print('pass1')

temp = None
while True:
    try:
        _, frame = cap.read()
        cv.imshow('Video', frame)
        frame = imutils.resize(frame, width=640)
        cv.imshow('Video_small', frame)
        print(frame.shape)
        if temp is None:
            temp = frame

        if cv.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        break
print(temp.shape[1])

temp = temp.reshape(temp.shape[0],-1)
temp = temp.reshape(temp.shape[0],-1)
np.savetxt('data/test.csv', temp)
load_temp = np.loadtxt('data/test.csv')

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
faces = detector(gray)
