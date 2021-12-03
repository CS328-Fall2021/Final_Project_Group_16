import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from features import FeatureExtractor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import pickle

# data should be named as eye-data-Chang-0.csv 
data_dir = 'data' # directory where the data files are stored

output_dir = 'training_output' # directory where the classifier(s) are stored

class_labels = ['blinking', 'eye open', 'eye close', 'frown']

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for filename in os.listdir(data_dir):
	if filename.endswith(".csv") and filename.startswith("eye-data"):
		filename_components = filename.split("-")
        tester = filename_components[2]
        print("Loading data for {}.".format(tester))
        speaker_label = class_labels.index(speaker)
        sys.stdout.flush()
        data_file = os.path.join(data_dir, filename)
        data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
