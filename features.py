import numpy as np
import cv2 as cv
import math, dlib
from scipy.signal import lfilter
from scipy.spatial import distance as dist


class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("content/shape_predictor_68_face_landmarks.dat")

    def _getHorizontalLineRatio(self, frames, HorizontalLength, which_eye):
        HorizontalLineRatio = []
        most_left_index = 36
        most_right_index = 39
        if which_eye == 'right':
            most_left_index = 42
            most_right_index = 45
        for frame in frames:
            horizontal_ratio = abs(frame[most_left_index][0] - frame[most_right_index][0])/HorizontalLength
            HorizontalLineRatio.append(horizontal_ratio)
        return HorizontalLineRatio

    def _getRatioMean(self,ratios):
        return np.mean(ratios)

    def _getRatioMedian(self,ratios):
        return np.median(ratios)

    def _getEyelength(self, frames, which_eye):
        # use histrgam to get most common interval and base on the range of it to determine the eye length when open

        most_left_index = 36  # eye_points = [[36, 39, 37, 38, 41, 40], [42, 45, 43, 44, 47, 46]]
        most_right_index = 39
        top_1_index = 37
        top_2_index = 38
        bottom_1_index = 41
        bottom_2_index = 40
        if which_eye == 'right':
            most_left_index = 42  # eye_points = [[36, 39, 37, 38, 41, 40], [42, 45, 43, 44, 47, 46]]
            most_right_index = 45
            top_1_index = 43
            top_2_index = 44
            bottom_1_index = 47
            bottom_2_index = 46

        Horizontal = []
        Vertical = []
        for frame in frames:
            Horizontal.append(abs(frame[most_left_index][0] - frame[most_right_index][0]))
            Vertical.append(abs((frame[top_1_index][1] + frame[top_2_index][1]) / 2 - (frame[bottom_2_index][1] + frame[bottom_1_index][1]) / 2))
        x_hist, x_range = np.histogram(Horizontal, bins=4)
        x_max = np.max(x_hist)
        x_result = np.where(x_hist == x_max)[0].tolist()[0]
        x_index = int(x_result)

        y_hist, y_range = np.histogram(Vertical, bins=4)
        y_max = np.max(y_hist)
        y_result = np.where(y_hist == y_max)[0].tolist()[0]
        y_index = int(y_result)

        x_length = (x_range[x_index] + x_range[x_index + 1]) / 2
        y_length = (y_range[y_index] + y_range[y_index + 1]) / 2

        return x_length, y_length

    def _getVerticalLineRatio(self, frames, verticalLength,which_eye):  # input 1 d array output 1d

        top_1_index = 37
        top_2_index = 38
        bottom_1_index = 41
        bottom_2_index = 40
        if which_eye == 'right':
            top_1_index = 43
            top_2_index = 44
            bottom_1_index = 47
            bottom_2_index = 46
        verticalLine_ratio = []
        for frame in frames:
            vertical_ratio = abs((frame[top_1_index][1] + frame[top_2_index][1]) / 2
                                   - (frame[bottom_2_index][1] + frame[bottom_1_index][1]) / 2)/ verticalLength
            verticalLine_ratio .append(vertical_ratio)
        return verticalLine_ratio

    # def _getLineRatio(self, frames, horizontalLength, VerticalLength):  # x y ratio v/h
    #     line_Ratio = []
    #     for frame in frames:
    #         vertical_length  = (abs(frame[36][0] - frame[39][0]))
    #         horizontal_length = (abs((frame[37][1] + frame[38][1]) / 2 - (frame[40][1] + frame[41][1]) / 2))
    #         line_Ratio.append(vertical_length/horizontal_length)
    #     return line_Ratio

    # def _getBothEyeRatio(self, frames): # frames is np array with (n # frame,68,2) shape
    #     # take the last frame of the window as data
    #     # return the left_ratio and right_ratio
    #
    #     left_ratio_lst = []
    #     right_ratio_lst = []
    #
    #     for frame in frames:
    #         # left eyes position tuple(x, y)
    #         index_36 = frame[36]
    #         index_37 = frame[37]
    #         index_38 = frame[38]
    #         index_39 = frame[39]
    #         index_40 = frame[40]
    #         index_41 = frame[41]
    #
    #         A_left = dist.euclidean(index_37, index_41)
    #         B_left = dist.euclidean(index_38, index_40)
    #         C_left = dist.euclidean(index_36, index_39)
    #         left_ratio = (A_left + B_left) / (2.0 * C_left)
    #         left_ratio_lst.append(left_ratio)
    #
    #         # right eyes position tuple(x, y)
    #         index_42 = frame[42]
    #         index_43 = frame[43]
    #         index_44 = frame[44]
    #         index_45 = frame[45]
    #         index_46 = frame[46]
    #         index_47 = frame[47]
    #
    #         A_right = dist.euclidean(index_43, index_47)
    #         B_right = dist.euclidean(index_44, index_46)
    #         C_right = dist.euclidean(index_42, index_45)
    #         right_ratio = (A_right + B_right) / (2.0 * C_right)
    #         right_ratio_lst.append(right_ratio)
    #     #apply histogram
    #     left_hist, left_range = np.histogram(left_ratio_lst, 4)
    #     left_range_min = left_range[1]
    #     left_range_max = left_range[-2]
    #
    #     right_hist, right_range = np.histogram(right_ratio_lst, 4)
    #     right_range_min = right_range[1]
    #     right_range_max = right_range[-2]
    #
    #     return left_range_min, left_range_max, right_range_min, right_range_max

    def _getBothEyeRatio(self, frames): # frames is np array with (n # frame,68,2) shape
        # take the last frame of the window as data
        # return the left_ratio and right_ratio

        len_frames = len(frames)
        frame = frames[len_frames-1]
        # left eyes position tuple(x, y)
        index_36 = frame[36]
        index_37 = frame[37]
        index_38 = frame[38]
        index_39 = frame[39]
        index_40 = frame[40]
        index_41 = frame[41]

        A_left = dist.euclidean(index_37, index_41)
        B_left = dist.euclidean(index_38, index_40)
        C_left = dist.euclidean(index_36, index_39)
        left_ratio = (A_left + B_left) / (2.0 * C_left)


        # right eyes position tuple(x, y)
        index_42 = frame[42]
        index_43 = frame[43]
        index_44 = frame[44]
        index_45 = frame[45]
        index_46 = frame[46]
        index_47 = frame[47]

        A_right = dist.euclidean(index_43, index_47)
        B_right = dist.euclidean(index_44, index_46)
        C_right = dist.euclidean(index_42, index_45)
        right_ratio = (A_right + B_right) / (2.0 * C_right)

        #print(f"Left_ratio: {left_ratio}")

        return left_ratio, right_ratio

    def _get_Eyes_ratio_variance(self, frames):
        ratio_lst = []
        for frame in frames:
            # left eyes position tuple(x, y)
            index_36 = frame[36]
            index_37 = frame[37]
            index_38 = frame[38]
            index_39 = frame[39]
            index_40 = frame[40]
            index_41 = frame[41]

            A_left = dist.euclidean(index_37, index_41)
            B_left = dist.euclidean(index_38, index_40)
            C_left = dist.euclidean(index_36, index_39)
            left_ratio = (A_left + B_left) / (2.0 * C_left)
            ratio_lst.append(left_ratio)

        mean = sum(ratio_lst) / len(ratio_lst)
        res = sum((i - mean) ** 2 for i in ratio_lst) / len(ratio_lst)

        #print(f"Eye_Ratio_Variance: {res}")
        return res

    def _get_Eyes_ratio_Difference(self, frames):
        ratio_lst = []
        for frame in frames:
            #print(f"frame: {frame}")
            # left eyes position tuple(x, y)
            index_36 = frame[36]
            index_37 = frame[37]
            index_38 = frame[38]
            index_39 = frame[39]
            index_40 = frame[40]
            index_41 = frame[41]

            A_left = dist.euclidean(index_37, index_41)
            B_left = dist.euclidean(index_38, index_40)
            C_left = dist.euclidean(index_36, index_39)
            left_ratio = (A_left + B_left) / (2.0 * C_left)
            ratio_lst.append(left_ratio)

        difference = max(ratio_lst) - min(ratio_lst)
        # print(ratio_lst)
        # print(f"max: {max(ratio_lst)}")
        # print(f"min: {min(ratio_lst)}")
        # print(f"Eye_Ratio_Difference: {difference}")
        return difference

    def _get_Eye_Vertical_Length(self, frames): # frames is np array with (n # frame,68,2) shape
        # take the last frame of the window as data
        # return the left_ratio and right_ratio

        len_frames = len(frames)
        frame = frames[len_frames-1]
        # left eyes position tuple(x, y)
        index_21 = frame[21][1]
        index_28 = frame[28][1]

        index_37 = frame[37][1]
        index_38 = frame[38][1]

        index_40 = frame[40][1]
        index_41 = frame[41][1]

        left_face_length = abs(index_21 - index_28)
        A_left = abs(index_37 - index_41)
        B_left = abs(index_38 - index_40)
        left_eye_ratio = ((A_left + B_left)/2)/left_face_length


        # right eyes position tuple(x, y)
        index_22 = frame[22][1]
        index_28 = frame[28][1]

        index_43 = frame[43][1]
        index_44 = frame[44][1]

        index_46 = frame[46][1]
        index_47 = frame[47][1]

        right_face_length = abs(index_22 - index_28)
        A_left = abs(index_43 - index_47)
        B_left = abs(index_44 - index_46)
        right_eye_ratio = ((A_left + B_left)/2)/right_face_length

        res = (left_eye_ratio + right_eye_ratio)/2

        #print(f"Eye_Vertical_Length: {res}")

        return res

    def extract_features(self, frames, debug=True):  # frames: 68 * n frames(time_domain)
        x = []
        y = []
        # left_HorizontalLength, left_VerticalLength = self._getEyelength(frames, 'left')
        # right_HorizontalLength, right_VerticalLength = self._getEyelength(frames, 'right')
        # x.append(left_VerticalLength)
        # y.append("left_vertical_Eye_Length")
        # x.append(left_HorizontalLength)
        # y.append("left_horizontal_Eye_Length")
        # x.append(right_VerticalLength)
        # y.append("right_vertical_Eye_Length")
        # x.append(right_HorizontalLength)
        # y.append("right_horizontal_Eye_Length")
        # left_x_ratio = self._getHorizontalLineRatio(frames, left_HorizontalLength, 'left')
        # left_y_ratio = self._getVerticalLineRatio(frames, left_VerticalLength, 'left')
        # right_x_ratio = self._getHorizontalLineRatio(frames, right_HorizontalLength, 'right')
        # right_y_ratio = self._getVerticalLineRatio(frames, right_VerticalLength, 'right')
        # x.append(self._getRatioMean(left_x_ratio))
        # y.append('left_vertical_Eye_Ratio_Mean')
        # x.append(self._getRatioMean(left_y_ratio))
        # y.append('left_horizontal_Eye_Ratio_Mean')
        # x.append(self._getRatioMean(right_x_ratio))
        # y.append('right_vertical_Eye_Ratio_Mean')
        # x.append(self._getRatioMean(right_y_ratio))
        # y.append('right_horizontal_Eye_Ratio_Mean')
        # x.append(self._getRatioMedian(left_x_ratio))
        # y.append('left_vertical_Eye_Ratio_Median')
        # x.append(self._getRatioMedian(left_y_ratio))
        # y.append('left_horizontal_Eye_Ratio_Median')
        # x.append(self._getRatioMedian(right_x_ratio))
        # y.append('right_vertical_Eye_Ratio_Median')
        # x.append(self._getRatioMedian(right_y_ratio))
        # y.append('right_horizontal_Eye_Ratio_Median')
        # x.append(self._getLineRatio(frames, HorizontalLength, VerticalLength))
        # y.append('VerticalToHorizontalRatio')
        # left_range_min, left_range_max, right_range_min, right_range_max= self._getBothEyeRatio(frames)
        # x.append(left_range_min)
        # y.append('left_eye_ratio_min')
        # x.append(right_range_min)
        # y.append('right_eye_ratio_min')
        # x.append(left_range_max)
        # y.append('left_eye_range_max')
        # x.append(right_range_max)
        # y.append('right_eye_range_max')
        left_ratio, right_ratio = self._getBothEyeRatio(frames)
        x.append(left_ratio)
        y.append('left_eye_ratio')
        x.append(right_ratio)
        y.append('right_eye_ratio')
        y_length = self._get_Eye_Vertical_Length(frames)
        x.append(y_length)
        y.append('Eyes_Vertical_Ratio')
        x.append(self._get_Eyes_ratio_Difference(frames))
        y.append('Left_Ratio_Difference')
        # if debug:
        #     print(x)
        #     print(y)
        #     print(len(x))
        #     print(len(y) == len(x))

        return x, y
