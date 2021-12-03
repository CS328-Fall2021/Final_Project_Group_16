import sys
import json
import os
import cv2 as cv
import numpy as np
import dlib


# class_labels = ['blinking', 'eye open', 'eye close', 'frown']
# labels_index = [    0     ,      1    ,       2    ,    3   ]
label = 0

filename="eye-data-Kai.csv"#"eye-data-HenryVIII-1.csv"

data_dir = "data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


cap = cv.VideoCapture(0)

while True:
    try:
        _, frame = cap.read()
        # each frame is a np array with shape 480*640*3
        


    except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Saving labelled data...")
            labelled_data = np.asarray(labelled_data)
            with open(os.path.join(data_dir, filename), "wb") as f:
                np.savetxt(f, labelled_data, delimiter=",")
            break

    except Exception as e:
        # ignore exceptions, such as parsing the json
        # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
        # but make sure the error is displayed so we know what's going on
        if (str(e) != "timed out"):  # ignore timeout exceptions completely       
            print(e)
        pass