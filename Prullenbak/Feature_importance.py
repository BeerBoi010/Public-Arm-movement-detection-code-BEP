### Importing of necessary libraries ###############################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2
from Feature_Extraction import Mean_V2
from Feature_Extraction import  Slope_V2
from Feature_Extraction import Max_V2
from Feature_Extraction import Min_V2
from Feature_Extraction import Standard_Deviation
from Random_forest import labels_interpolation


##### VARIABLES ######################################################################################################
'''later toevoegen dat random wordt gekozen wie train en test is'''

train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 2
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices


subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
        'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']


# subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']
# #print(subjects_test)

test_labels = all_labels[test_person - 2]
# #print("test labels:",test_labels)

# all_labels.pop(test_person - 2)
train_labels = all_labels
# #print("train labels:",train_labels)

#################################################################################################################
### Setting up the test and training sets with labels ###########################################################

X_train_RMS = RMS_V2.RMS_train(subjects_train, sampling_window_RMS, min_periods)
X_test_RMS = RMS_V2.RMS_test(subjects_test, sampling_window_RMS, min_periods)

X_train_Mean = Mean_V2.Mean_train(subjects_train, sampling_window_mean, min_periods)
X_test_Mean = Mean_V2.Mean_test(subjects_test, sampling_window_mean, min_periods)

X_train_Slope = Slope_V2.Slope_train(subjects_train, sampling_window_slope, min_periods)
X_test_Slope = Slope_V2.Slope_test(subjects_test, sampling_window_slope, min_periods)

X_train_Max = Max_V2.Max_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Max = Max_V2.Max_test(subjects_test, sampling_window_min_max, min_periods)

X_train_Min = Min_V2.Min_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Min = Min_V2.Min_test(subjects_test, sampling_window_min_max, min_periods)

X_train_STD = Standard_Deviation.STD_train(subjects_train, sampling_window_STD, min_periods)
X_test_STD = Standard_Deviation.STD_test(subjects_test, sampling_window_STD, min_periods)

Y_train_labels = train_labels
Y_test_labels = test_labels


labels_train = []
###### for-loops to make annotation list for random forest method ###########################################################################
for item in Y_train_labels:
    for i in item:
        labels_train.append(i[1])

labels_test = []

for item in Y_test_labels:
    labels_test.append(item[1])


# Dictionary to map labels to numerical values
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Convert labels to numerical values
y_train = [label_mapping[label] for label in labels_train]
y_test = [label_mapping[label] for label in labels_test]

#print("y_test",len(y_test))

#### Create lists to store test and train data and labels for each patient #################################################################

'''For-loop below makes separate arrays for different subjects and different sensors, 
    with two different arrays for every feature'''

X_data_patients_train = []

for subject in X_train_RMS:

    # Initialize combined_data_patient for each patient
    combined_data_patient = []

    # Combine accelerometer and gyroscope data horizontally
    for imu_location in X_train_RMS[subject]:

        acc_rms_imu = X_train_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_train_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_train_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_train_Mean[subject][imu_location]["rot_mean"]
        acc_slope_imu = X_train_Slope[subject][imu_location]["acc_slope"]
        rot_slope_imu = X_train_Slope[subject][imu_location]["rot_slope"]
        acc_max_imu = X_train_Max[subject][imu_location]["acc_max"]
        rot_max_imu = X_train_Max[subject][imu_location]["rot_max"]
        acc_min_imu = X_train_Min[subject][imu_location]["acc_min"]
        rot_min_imu = X_train_Min[subject][imu_location]["rot_min"]
        acc_STD_imu = X_train_STD[subject][imu_location]["acc_STD"]
        rot_STD_imu = X_train_STD[subject][imu_location]["rot_STD"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu,acc_slope_imu,rot_slope_imu,
                                       acc_max_imu,rot_max_imu,acc_min_imu,rot_min_imu,acc_STD_imu,rot_STD_imu))
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

'''Arrays for all combined train data'''
combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train
shape = X_train.shape
# print(shape)
# print(X_train)
#print(combined_X_data_train.shape)
#############################################################################################################################
###### Arrays for test data ################################################################################################

'''For-loop below makes separate arrays for different subjects and different sensors, 
    with two different arrays for every feature'''


#print(combined_X_data_test.shape) ##test print to see the general shape

########################################################################################################################
################ RANDOM FOREST CLASSIFIER ##############################################################################

'''  Below all parameters are set up to run the random forest classifier '''

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


##########################################################################################################################
#### Plots for visualization #############################################################################################

'''Below plots are made to visualize what the Random classifier has done and how it has performed'''


#plt.show()
# Get feature importances
importances = clf.feature_importances_
# Number of top important features to select
n = 180

# Get indices of top n important features
top_indices = np.argsort(importances)[::-1][:n]
print(importances)
print(top_indices)
# Select only top n important features from the original feature matrix
X_train_selected = X_train[:, top_indices]

# # Optionally, you can print the indices of the selected features
print("Indices of selected features:", top_indices)

# Shape of the new feature matrix
print("Shape of new feature matrix:", X_train_selected.shape)



#Original feature numbers (0 - 179) are preserved.

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()
