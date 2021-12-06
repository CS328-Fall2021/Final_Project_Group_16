import os
import dlib
import cv2 as cv

WINDOW_SIZE = 5 # how many frames in each window

FONT = cv.FONT_HERSHEY_SIMPLEX

class_labels = ['eye open', 'blinking', 'frown']
# class_labels = ['eye open', 'blinking', 'frown']
labels_index = [    0     ,      1    ,    2   ]

eye_points = [[36,39,37,38,41,40],[42,45,43,44,47,46]]

data_dir = 'data' # directory where the data files are stored
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

output_dir = 'training_output' # directory where the classifier(s) are stored
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")

class NoLabelDetectedIntheFrame(Exception):
    pass