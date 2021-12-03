
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

    def _getVerticalLineRatio(self,landmarks):
        pass
    def _getEyelength(self,window,landmarks): # use histrgam to get most common interval and base on the range of it to determine the eye length when open
        pass

    def _getHorizontalLineRatio(self,window,landmarks):  #input 1 d array output 1d

        pass
    
    def _getLineRatio(self,landmarks): # x y ratio v/h
        pass
    # frame is np array with (480,640,3) shape
    def extract_features(self, frame, debug=True):
        x = []
        y = []
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self._getEyePoints(gray, face)
            x.append(landmarks)
            y.append('EyePoints')
        return x,y