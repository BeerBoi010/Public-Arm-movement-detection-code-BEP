import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns


from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation


train_amount = 6
sampling_window = 3
min_periods = 1
test_amount = train_amount

sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3

test_person = 7

print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
train_labels = all_labels

all_window_features = []
full_data = []

num_features = 30
# Initialize Feature_Matrix as an empty array with the correct number of columns
Feature_Matrix = np.empty((0, num_features))
imu_list  = np.empty((0, num_features))
Full_feature_split = np.empty((0, num_features))
for subject in subjects:
    all_window_features = []

    for imu_location in acc[subject]:
        # Splitting the data into 381 segments
        acc_data_imu_split = np.array_split(acc[subject][imu_location], 381)
        rot_data_imu_split = np.array_split(rot[subject][imu_location], 381)

        # Concatenate acc and rot data for each segment
        full_data = [np.concatenate((acc, rot), axis=1) for acc, rot in zip(acc_data_imu_split, rot_data_imu_split)]

        for split in full_data:
            # Calculate the features for each segment
            Mean = np.mean(split, axis=0)
            STD = np.std(split, axis=0)
            RMS = np.sqrt(np.mean(split**2, axis=0))
            MIN = np.min(split, axis=0)
            MAX = np.max(split, axis=0)

            # Ensure features are arrays before concatenation
            Mean = np.atleast_1d(Mean)
            STD = np.atleast_1d(STD)
            RMS = np.atleast_1d(RMS)
            MIN = np.atleast_1d(MIN)
            MAX = np.atleast_1d(MAX)

            # Concatenate features into a single array
            window_features = np.concatenate([Mean, STD, RMS, MIN, MAX])
            # Append the features for the current window to all_window_features

        #Here we should implement that all imu locations should stack the columns to creat 180 or 150 features


    # Convert the list of all window features to a NumPy array
    subject_feature_matrix = np.array(all_window_features)
    print(subject_feature_matrix.shape)

    # Append the features for the current subject to Feature_Matrix
    Feature_Matrix = np.vstack((Feature_Matrix, subject_feature_matrix))

print('Final Feature Matrix shape: ', Feature_Matrix.shape)
print('Final Feature Matrix: \n', Feature_Matrix)