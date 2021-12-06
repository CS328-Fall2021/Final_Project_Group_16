import os
import dlib
import cv2 as cv

WINDOW_SIZE = 15 # how many frames in each window

FONT = cv.FONT_HERSHEY_SIMPLEX

class_labels = ['Eye Open', 'Blinking', 'Frown', 'Staring']
# class_labels = ['eye open', 'blinking', 'frown']
labels_index = [    0     ,     1     ,    2   ,     3    ]

eye_points = [[36,39,37,38,41,40],[42,45,43,44,47,46]]

data_dir = 'data' # directory where the data files are stored
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

output_dir = 'training_output' # directory where the classifier(s) are stored
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")


def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def draw_eye(landmarks, frame, points):
    left_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    right_point = (landmarks.part(points[1]).x, landmarks.part(points[1]).y)
    center_top = midpoint(landmarks.part(points[2]), landmarks.part(points[3]))
    center_bottom = midpoint(landmarks.part(points[4]), landmarks.part(points[5]))

    hor_line = cv.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    return frame

class NoLabelDetectedIntheFrame(Exception):
    pass