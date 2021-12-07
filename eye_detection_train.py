import os, sys, pickle, math
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from timeit import default_timer as timer
from features import FeatureExtractor
from utils import data_dir, output_dir, class_labels, WINDOW_SIZE, detector, predictor, NoLabelDetectedIntheFrame

# csv should be named as eye-data-Chang-0.csv

class_names = []

data = []
classifier_filename = 'classifier.pickle'

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("eye-data"):
        filename_components = filename.split("-")
        label_index = int(filename_components[2].strip('.csv'))
        eye_label = class_labels[label_index]
        print("\nLoading data for '{}'.".format(eye_label))
        if eye_label not in class_names:
            class_names.append(eye_label)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        raw_data_for_current_label = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled frame samples.".format(len(raw_data_for_current_label)))
        size_of_raw_data = len(raw_data_for_current_label)
        cur_data_size = len(data)
        for i in range(math.ceil(size_of_raw_data / WINDOW_SIZE)):
            if (i + 1) * 20 > size_of_raw_data:
                data.append(raw_data_for_current_label[i * WINDOW_SIZE:])
            else:
                data.append(raw_data_for_current_label[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE])

        print("Split {} raw labelled frame samples into {} windows.".format(len(raw_data_for_current_label),
                                                                            len(data) - cur_data_size))

        sys.stdout.flush()

print("Found data for {} label : {}".format(len(class_names), ", ".join(class_names)))

n_features = 16

print("\nExtracting features and labels for {} windows...".format(len(data)))
print('Window Size: {}'.format(WINDOW_SIZE))
sys.stdout.flush()

X = np.zeros((0, n_features))
y = np.zeros(0, )

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False)

# iterating each window which contains atmost 20 frames
# data is list in format [frames, frames, ....]
# frames is list in format [frame, frame, ....]
# frame is np array with shape (68*2+1)*1  
print('Timer started...')
extract_start = timer()

for i, window_frames_with_label in enumerate(data):
    frames_three_dimention_data = []
    label = None

    for frame_one_dimention_data in window_frames_with_label:
        if label is None: label = frame_one_dimention_data[-1]
        unraveled = frame_one_dimention_data[:-1].copy().reshape([68, 2])
        frames_three_dimention_data.append(unraveled)

    if label is None: raise NoLabelDetectedIntheFrame
    # print(frames_three_dimention_data[0])

    # window would be WINDOW_SIZE*68*2 for x,y coordinate for 68 points for WINDOW_SIZE numbers of frames
    x, features_names = feature_extractor.extract_features(frames_three_dimention_data)
    if len(x) != X.shape[1]:
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
    X = np.append(X, np.reshape(x, (1, -1)), axis=0)
    y = np.append(y, label)

extract_end = timer()
print("Extracted features in {:.1f} minutes".format((extract_end-extract_start)/60))
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

n = len(y)
n_classes = len(class_names)

print("\n")
print("---------------------- Decision Tree -------------------------")
#		                Train & Evaluate Classifier

cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
for index, (train_index, test_index) in enumerate(cv.split(X)):
    tree = DecisionTreeClassifier(criterion="entropy")
    #print("Train: ", train_index, "Test: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tree.fit(X_train, y_train)
    y_pre = tree.predict(X_test)
    conf = confusion_matrix(y_test, y_pre)
#     print('\nFold: {:>2}'.format(index+1))
#     print('Confusion Matrix:')
#     print(conf)
#     print('Average Accuracy: ', tree.score(X_test, y_test))
#     print('Precision Value: ', precision_score(y_test, y_pre, average='micro'))
#     print('Recall Value: ', recall_score(y_test, y_pre, average='micro'))

tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X, y)
export_graphviz(tree, out_file=os.path.join(output_dir, 'sample.dot'), feature_names=features_names, class_names=class_names)
with open(os.path.join(output_dir, classifier_filename), 'wb') as f:
    pickle.dump(tree, f)
f.close()
