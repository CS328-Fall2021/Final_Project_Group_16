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

    def _getVerticalLineRatio(self, frames, VerticalLength):
        VerticalLineRatio = []
        for frame in frames:
            vertial_ratio = abs(frame[36][0] - frame[39][0])/VerticalLength
            VerticalLineRatio.append(vertial_ratio)
        return VerticalLineRatio

    def _getEyelength(self,frames):
        # use histrgam to get most common interval and base on the range of it to determine the eye length when open
        Horizontal = []
        Vertical = []
        for frame in frames:
            Vertical.append(abs(frame[36][0] - frame[39][0]))
            Horizontal.append(abs((frame[37][1] + frame[38][1]) / 2 - (frame[40][1] + frame[41][1]) / 2))
        x_hist, x_range = np.histogram(Vertical, bins=4)
        x_max = np.max(x_hist)
        x_result = np.where(x_hist == x_max)[0].toList()[0]
        x_index = int(x_result)

        y_hist, y_range = np.histogram(Horizontal, bins=4)
        y_max = np.max(y_hist)
        y_result = np.where(y_hist == y_max)[0].toList()[0]
        y_index = int(y_result)

        x_length = (x_range[x_index] + x_range[x_index + 1]) / 2
        y_length = (y_range[y_index] + y_range[y_index + 1]) / 2

        return [x_length], [y_length]

    def _getHorizontalLineRatio(self, frames, horizontalLength):  # input 1 d array output 1d
        HorizontalLineRatio = []
        for frame in frames:
            horizontal_ratio = abs((frame[37][1] + frame[38][1]) / 2 - (frame[40][1] + frame[41][1]) / 2) / horizontalLength
            HorizontalLineRatio .append(horizontal_ratio)
        return HorizontalLineRatio 

    def _getLineRatio(self, frames, horizontalLength, VerticalLength):  # x y ratio v/h
        line_Ratio = []
        for frame in frames:
            vertical_length  = (abs(frame[36][0] - frame[39][0]))
            horizontal_length = (abs((frame[37][1] + frame[38][1]) / 2 - (frame[40][1] + frame[41][1]) / 2))
            line_Ratio.append(vertical_length/horizontal_length)
        return line_Ratio
    # frame is np array with (480,640,3) shape
    def extract_features(self, frames, debug=True):  # frames: 68 * n frames(time_domain)
        x = []
        y = []
        VerticalLength, HorizontalLength = self._getEyelength(frames)
        x.append(HorizontalLength)
        y.append("vertical_Eye_Length")
        x.append(VerticalLength)
        y.append("horizontal_Eye_Length")

        return x, y
