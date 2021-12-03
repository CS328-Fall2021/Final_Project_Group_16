import cv2 as cv
import numpy as np
import dlib, os, imutils

class NotEqual(Exception):
    pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")

label = 0

filename="eye-data-Chang.csv"#"eye-data-Chang-0.csv"

data_dir = "data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

labelled_data = []

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

temp = None
while True:
    try:
        _, frame = cap.read()
        # cv.imshow('Video', frame)
        frame = imutils.resize(frame, width=640)
        cv.imshow('Video_small', frame)
        # raveled = np.append(frame.ravel(),label)
        # labelled_data.append(raveled)

        if temp is None:
            temp = frame
            gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                temp = None
                continue
            print(len(faces))
            landmarks = predictor(gray, faces[0])
            face_points = []
            for point in landmarks.parts():
                face_points.append([point.x, point.y])
            print(np.asarray(face_points).flatten())
            # print(landmarks.part(0).x)
            # print(landmarks.parts()[0])

        if cv.waitKey(1) == ord('q'):
            raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("User Interrupt. Saving labelled data...")
        # # labelled_data = np.asarray(labelled_data)
        # raveled = np.append(temp.ravel(),label)
        # labelled_data.append(raveled)
        # with open(os.path.join(data_dir, filename), "wb") as doc:
        #     np.savetxt(doc, labelled_data, delimiter=",")
        break


# load_temp = np.genfromtxt('data/eye-data-Chang.csv', delimiter=',')
# print(len(load_temp))
# data = load_temp
# label = data[-1]
# unraveled = data[:-1].copy().reshape([480,640,3])
# print(unraveled.shape)



# print(raveled.shape)
# np.savetxt('data/test.csv', raveled)
# # load_temp = np.loadtxt('data/test.csv')

# 

# for data in load_temp:

#     label = data[-1]
#     unraveled = data[:-1].copy().reshape([480,640,3])
#     print('label is: ', label)
#     # if (temp == unraveled).all():
#     #     print('True')
#     # else: raise NotEqual
#     faces = detector(gray)
#     print(len(faces))

# import os
# import sys
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from features import FeatureExtractor
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# import pickle

# data_dir = 'data' # directory where the data files are stored

# output_dir = 'training_output' # directory where the classifier(s) are stored

# class_labels = ['blinking', 'eye open', 'frown']

# if not os.path.exists(output_dir):
# 	os.mkdir(output_dir)

# for filename in os.listdir(data_dir):
# 	if filename.endswith(".csv") and filename.startswith("eye-data"):
# 		filename_components = filename.split("-")
#         tester = filename_components[2]
#         print("Loading data for {}.".format(tester))