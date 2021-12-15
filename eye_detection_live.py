import cv2 as cv
import dlib
import numpy as np
import pickle
from features import FeatureExtractor
from utils import FONT, WINDOW_SIZE, class_labels, labels_index, output_dir, eye_points, detector, predictor, midpoint, draw_eye
import os, sys, random


classifier_filename = 'classifier.pickle'
debug = False

if not debug:
    with open(os.path.join(output_dir, classifier_filename), 'rb') as f:
        classifier = pickle.load(f)
        
    if classifier == None:
        print("Classifier is null; make sure you have trained it!")
        sys.exit()
    
feature_extractor = FeatureExtractor(debug=False)
cap = cv.VideoCapture(0, cv.CAP_DSHOW) #  use(0, cv.CAP_DSHOW) for windows and (0) for mac

def print_activity(frame, activity_str):
    cv.putText(frame, activity_str, (50,150), FONT, 3, (255,0,0), thickness=3)
    return frame


def ActivityDetected(frame, activity):
    """
    Notifies the user of the current speaker
    """
    print_activity(frame, class_labels[activity])
    print("Current activity: {}.".format(class_labels[activity]))
    sys.stdout.flush()
    return frame

def predict(window):
    
    X, feature_names = feature_extractor.extract_features(window)
    X = np.reshape(X,(1,-1))
    if not debug:
        index = classifier.predict(X) 
    else:
        index = random.choice(labels_index)
    return int(index)


try:
    cur_label = 'No label is predicted yet.'
    notready = True
    cur_samples = []
    while True:
        _, frame = cap.read()
        
        if notready:
            cv.putText(frame, 'Press R When You Ready', (50,150), FONT, 1, (255,0,0), thickness=3)
            cv.imshow("Eye Movement Classification", frame)

            waited = cv.waitKey(1)
            if waited == ord('r'): notready = False
            if waited == ord('q'): raise KeyboardInterrupt
            continue
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) == 1:

            landmarks = predictor(gray, faces[0])
            face_points = []
            for point in landmarks.parts():
                face_points.append([point.x, point.y])
            raveled = np.asarray(face_points)
            cur_samples.append(raveled)
            
            frame = draw_eye(landmarks, frame, eye_points[0])
            frame = draw_eye(landmarks, frame, eye_points[1])

            if len(cur_samples) >= WINDOW_SIZE:
                activity_index = predict(cur_samples)
                cur_label = class_labels[activity_index]
                frame = ActivityDetected(frame, activity_index) 
                cur_samples.clear()
            else:
                frame = print_activity(frame, cur_label) 


        cv.imshow("Eye Movement Classification", frame)


        waited = cv.waitKey(1)
        if waited == ord('p'): 
            notready = True
            print('User Paused...')
        if waited == ord('q'): raise KeyboardInterrupt
            
except KeyboardInterrupt:
    print('User Keyboard Interrupt...')
# except Exception as e:
#     print(e)
finally:
    cap.release()
    cv.destroyAllWindows()