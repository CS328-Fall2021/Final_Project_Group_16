import cv2 as cv
import numpy as np
import dlib,os
from utils import draw_eye, midpoint, eye_points


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        face_points = []
        for point in landmarks.parts():
            face_points.append([point.x, point.y])
        raveled = np.asarray(face_points)
        print(raveled)
        frame = draw_eye(landmarks, frame, eye_points[0])
        frame = draw_eye(landmarks, frame, eye_points[1])

    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()