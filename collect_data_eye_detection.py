import os
import cv2 as cv
import numpy as np
import imutils, dlib
from utils import data_dir, detector, predictor, FONT


# class_labels = ['eye open', 'blinking', 'frown']
# labels_index = [    0     ,      1    ,    2   ]

label = 0

filename="eye-data-{}.csv".format(label)#"eye-data-1.csv"

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

raw_data = []

try:
    notready = True
    while True:
        try:
            _, frame = cap.read()
            if notready:
                cv.putText(frame, 'Press R When You Ready', (50,150), FONT, 1, (255,0,0))
                cv.imshow("Eye Movement Classification", frame)

                if cv.waitKey(1) == ord('r'): notready = False
                if cv.waitKey(1) == ord('q'): raise KeyboardInterrupt
                continue
            # each frame is a np array with shape 480*640*3
            frame = imutils.resize(frame, width=640)
            cv.imshow('Video_small', frame)
            raw_data.append(frame)      

            if cv.waitKey(1) == ord('p'): notready = True
            if cv.waitKey(1) == ord('q'):
                raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("User Interrupt. Detecting Faces in the Frame and Extracting landmarks on Faces...\n")
            
            labelled_data = []
            for data in raw_data:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = detector(gray)

                if len(faces) == 1:
                    landmarks = predictor(gray, faces[0])
                    face_points = []
                    for point in landmarks.parts():
                        face_points.append([point.x, point.y])
                    raveled = np.append(np.asarray(face_points).flatten(), label)
                    labelled_data.append(raveled)

                # each row would contain (68*2+1,) 
                # which is (face has 68 face landmarks elements) + 1 label 
                # default the number of face is 1

            print('Extraction finished. Saving labelled data...\n')
            labelled_data = np.asarray(labelled_data)
            with open(os.path.join(data_dir, filename), "wb") as doc:
                np.savetxt(doc, labelled_data, delimiter=",")
            
            print('Labelled data saved...\n')
            break

        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass
except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
    quit()

finally:
    print('Closing Video Data Capturing')
    cap.release()
    cv.destroyAllWindows()