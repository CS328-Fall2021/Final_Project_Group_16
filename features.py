
import numpy as np
import math
from scipy.signal import lfilter
from audiolazy import lpc
from python_speech_features import mfcc

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug

    def _getEyePoints(self):
        pass

    def _getVerticalLine(self):
        pass
    
    def _getHorizontalLine(self):
        pass
    
    def _getLineRatio():
        pass

    def extract_features(self, window, debug=True):
        x = []

        return x