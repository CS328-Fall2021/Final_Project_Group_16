
import numpy as np
import cv2 as cv
import math, dlib
from scipy.signal import lfilter
from audiolazy import lpc
from python_speech_features import mfcc

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")


    def _getEyePoints(self, gray, face):
        return self.predictor(gray, face)

    def _getVerticalLine(self,landmarks):
        pass
    
    def _getHorizontalLine(self,landmarks):
        pass
    
    def _getLineRatio(self,landmarks):
        pass

    # frame is np array with (480,640,3) shape
    def extract_features(self, frame, debug=True):
        X = []
        y = []
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self._getEyePoints(gray, face)
            X.append(landmarks)
            y.append('EyePoints')
        return X,y