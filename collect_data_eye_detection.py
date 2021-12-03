import os
import cv2 as cv
import numpy as np
import imutils


# class_labels = ['blinking', 'eye open', 'frown']
# labels_index = [    0     ,      1    ,    2   ]

label = 0

filename="eye-data-Kai.csv"#"eye-data-HenryVIII-1.csv"

data_dir = "data"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


cap = cv.VideoCapture(0)

labelled_data = []

try:
    while True:
        try:
            _, frame = cap.read()
            # each frame is a np array with shape 480*640*3
            frame = imutils.resize(frame, width=640)
            cv.imshow('Video_small', frame)
            raveled = np.append(frame.ravel(),label)
            # each row would contain (921601,) which 480*640*3 video elements + 1 label 
            labelled_data.append(raveled)

            if cv.waitKey(1) == ord('q'):
                raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("User Interrupt. Saving labelled data...")
            labelled_data = np.asarray(labelled_data)
            with open(os.path.join(data_dir, filename), "wb") as doc:
                np.savetxt(doc, labelled_data, delimiter=",")
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