import cv2 as cv
import numpy as np
import dlib,os


cap = cv.VideoCapture(0, cv.CAP_DSHOW)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

eye_points = [[36,39,37,38,41,40],[42,45,43,44,47,46]]

def draw_eye(landmarks, frame, points):
    left_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    right_point = (landmarks.part(points[1]).x, landmarks.part(points[1]).y)
    center_top = midpoint(landmarks.part(points[2]), landmarks.part(points[3]))
    center_bottom = midpoint(landmarks.part(points[4]), landmarks.part(points[5]))

    hor_line = cv.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return frame

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