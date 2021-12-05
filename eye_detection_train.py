import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from features import FeatureExtractor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pickle
import math
from utils import data_dir, output_dir, class_labels, WINDOW_SIZE, detector, predictor, NoLabelDetectedIntheFrame



# csv should be named as eye-data-Chang-0.csv 

class_names = []



data = []

for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("eye-data"):
        filename_components = filename.split("-")
        label_index = int(filename_components[2])
        eye_label = class_labels[label_index]
        print("Loading data for {}.".format(eye_label))
        if eye_label not in class_names:
            class_names.append(eye_label)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        raw_data_for_current_label = np.genfromtxt(data_file, delimiter=',')
        print("Loaded {} raw labelled frame samples.".format(len(raw_data_for_current_label)))
        size_of_raw_data = len(raw_data_for_current_label)
        cur_data_size = len(data)
        for i in range(math.ceil(size_of_raw_data/WINDOW_SIZE)):
            if (i+1)*20 > size_of_raw_data:
                data.append(raw_data_for_current_label[i*WINDOW_SIZE:])
            else:
                data.append(raw_data_for_current_label[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE])

        print("Split {} raw labelled frame samples into {} windows.".format(len(raw_data_for_current_label), len(data)-cur_data_size))

        sys.stdout.flush()
        

print("Found data for {} label : {}".format(len(class_names), ", ".join(class_names)))


n_features = 12

print("Extracting features and labels for {} windows...".format(len(data)))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False) 

# iterating each window which contains atmost 20 frames
# data is list in format [frames, frames, ....]
# frames is list in format [frame, frame, ....]
# frame is np array with shape (68*2+1)*1  
for i, window_frames_with_label in enumerate(data):
    frames_three_dimention_data = []
    label = None

    for frame_one_dimention_data in window_frames_with_label:
        if label is None: label = frame_one_dimention_data[-1]
        unraveled = frame_one_dimention_data[:-1].copy().reshape([68,2])
        frames_three_dimention_data.append(unraveled)
	
    if label is None: raise NoLabelDetectedIntheFrame
    # print(frames_three_dimention_data[0])
    
    # window would be WINDOW_SIZE*68*2 for x,y coordinate for 68 points for WINDOW_SIZE numbers of frames
    x,y = feature_extractor.extract_features(frames_three_dimention_data)
    if (len(x) != X.shape[1]):
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    y = np.append(y, label)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()


n = len(y)
n_classes = len(class_names)


print("\n")
print("---------------------- Decision Tree -------------------------")


total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]

cv = KFold(n_splits=2, shuffle=True, random_state=None)
for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
	print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	tree.fit(X_train, y_train)
	print("Evaluating classifier over {} points...".format(len(y_test)))

	# predict the labels on the test data
	y_pred = tree.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

print("Training decision tree classifier on entire dataset...")
tree.fit(X, y)

print("\n")
print("---------------------- Random Forest Classifier -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	print("Evaluating classifier over {} points...".format(len(y_test)))
	# predict the labels on the test data
	y_pred = clf.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

# TODO: (optional) train other classifiers and print the average metrics using 10-fold cross-validation

# Set this to the best model you found, trained on all the data:
best_classifier = RandomForestClassifier(n_estimators=100)
best_classifier.fit(X,y) 

classifier_filename='classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
	pickle.dump(best_classifier, f)