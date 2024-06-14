#description: implemented features for the model to train on 


### Importing of necessary libraries ###############################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA

#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2
from Feature_Extraction import Mean_V2
from Feature_Extraction import  Slope_V2
from Feature_Extraction import Max_V2
from Feature_Extraction import Min_V2
from Feature_Extraction import Standard_Deviation
# from Feature_Extraction import entropy_V2
import labels_interpolation


##### VARIABLES ######################################################################################################
'''later toevoegen dat random wordt gekozen wie train en test is'''

sampling_window = 3
min_periods = 1
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
sampling_window_entropy = 3
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
# subjects_train = subjects
# subjects_test = [f'drinking_HealthySubject{test_person}_Test']
# print(subjects_test)

# test_labels = all_labels[test_person - 2]
# #print("test labels:",test_labels)

# all_labels.pop(test_person - 2)
# train_labels = all_labels
# #print("train labels:",train_labels)
#1524 381
important = [34, 8, 35, 5, 36, 33, 26, 20, 41, 31, 44, 56, 29, 62, 59, 30, 69, 23,
        32, 37, 65, 0, 47, 11, 4, 134, 3, 39, 6, 116, 67, 24, 71, 54, 18, 60,
        19, 42, 43, 25, 128, 113, 90, 70, 115, 7, 133, 68, 109, 77, 28, 127, 
        83, 27, 61, 22, 92, 95, 101, 78, 72, 96, 55, 10, 80, 147, 91, 98, 38,
        2, 21, 132, 66, 114, 126, 63, 97, 1, 152, 131, 170, 58, 40, 137, 130,
        57, 45, 9, 46, 79, 73, 107, 76, 164, 111, 117, 118, 100, 166, 173, 
        149, 172, 64, 154, 16, 74, 75, 119, 136, 165, 53, 148, 94, 135, 153, 
        167, 82, 143, 129, 171, 155, 151, 146, 169, 110, 112, 14, 99, 108, 
        17, 163, 93, 15, 125, 138, 162, 145, 168, 81, 103, 52, 89, 12, 177, 
        150, 142, 144, 50, 179, 13, 140, 105, 104, 102, 48, 51, 175, 174, 141, 
        106, 139, 123, 159, 176, 178, 161, 124, 85, 88, 49, 120, 84, 160, 86, 
        122, 121, 157, 156, 87, 158]
# Number of top important features to select
n = 30
# Get indices of top n important features
top_indices = important[:n]
subject = 7
split = 0.8
split1 = int(split * 1905)
subjects_train = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
         'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']
subjects_test = [f'drinking_HealthySubject{subject}_Test']
train_labels = all_labels
test_labels = all_labels[subject -2]
# label = all_labels[subject - 2]
# train_labels = label[:1523]
# test_labels = label[1524:]

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

# X_train_entropy = entropy_V2.Entropy_train(subjects_train, sampling_window_entropy)
# X_test_entropy = entropy_V2.Entropy_test(subjects_test, sampling_window_entropy)

Y_train_labels = train_labels
Y_test_labels = test_labels
# print("test labels", test_labels)

labels_train = []
###### for-loops to make annotation list for random forest method ###########################################################################
for item in Y_train_labels:
    for i in item[:split1]:
        labels_train.append(i[1])
# print("labels train", labels_train)

labels_test1 = []
for item in Y_test_labels:
    labels_test1.append(item[1])
labels_test = labels_test1[split1:]
# print("labels test", labels_test)

# Dictionary to map labels to numerical values
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Convert labels to numerical values
y_train = [label_mapping[label] for label in labels_train]
y_test = [label_mapping[label] for label in labels_test]

print("y_train",len(y_train))
print("y_test",len(y_test))

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
        # acc_entropy_imu = X_train_entropy[subject][imu_location]["acc_entropy"]
        # rot_entropy_imu = X_train_entropy[subject][imu_location]["rot_entropy"]

        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu,acc_slope_imu,rot_slope_imu,
                                       acc_max_imu,rot_max_imu,acc_min_imu,rot_min_imu,acc_STD_imu,rot_STD_imu))
        # ,acc_entropy_imu,rot_entropy_imu
        combined_data_patient.append(combined_data_imu[:split1])  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

'''Arrays for all combined train data'''
combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train[:, top_indices]
#X_train = combined_X_data_train
print("X_train_shape", X_train.shape)
#############################################################################################################################
###### Arrays for test data ################################################################################################

'''For-loop below makes separate arrays for different subjects and different sensors, 
    with two different arrays for every feature'''

X_data_patients_test = []

for subject in X_test_RMS:
    print("test subject", subject)
    # Initialize combined_data_patient for each patient
    combined_data_patient = []

    # Combine accelerometer and gyroscope data horizontally
    for imu_location in X_test_RMS[subject]:

        acc_rms_imu = X_test_RMS[subject][imu_location]["acc_rms"]
        rot_rms_imu = X_test_RMS[subject][imu_location]["rot_rms"]
        acc_mean_imu = X_test_Mean[subject][imu_location]["acc_mean"]
        rot_mean_imu = X_test_Mean[subject][imu_location]["rot_mean"]
        acc_slope_imu = X_test_Slope[subject][imu_location]["acc_slope"]
        rot_slope_imu = X_test_Slope[subject][imu_location]["rot_slope"]
        acc_max_imu = X_test_Max[subject][imu_location]["acc_max"]
        rot_max_imu = X_test_Max[subject][imu_location]["rot_max"]
        acc_min_imu = X_test_Min[subject][imu_location]["acc_min"]
        rot_min_imu = X_test_Min[subject][imu_location]["rot_min"]
        acc_STD_imu = X_test_STD[subject][imu_location]["acc_STD"]
        rot_STD_imu = X_test_STD[subject][imu_location]["rot_STD"]
        # acc_entropy_imu = X_test_entropy[subject][imu_location]["acc_entropy"]
        # rot_entropy_imu = X_test_entropy[subject][imu_location]["rot_entropy"]


        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu,acc_slope_imu,rot_slope_imu,
                                       acc_max_imu,rot_max_imu,acc_min_imu,rot_min_imu,acc_STD_imu,rot_STD_imu))
        # ,acc_entropy_imu,rot_entropy_imu
        combined_data_patient.append(combined_data_imu[split1:])  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_test.append(np.hstack(combined_data_patient))

'''Combine data from all patients'''
combined_X_data_test = np.concatenate(X_data_patients_test)
X_test = combined_X_data_test[:, top_indices]
# X_test = combined_X_data_test

print("X_test_shape", X_test.shape) ##test print to see the general shape

########################################################################################################################
################ RANDOM FOREST CLASSIFIER ##############################################################################

'''  Below all parameters are set up to run the random forest classifier '''

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
print("y_test_pred",len(y_test_pred))
y_train_pred = clf.predict(X_train)

# Display classification report  of training data
print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

# Display classification report of test data
print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

 # Create an empty list of size equal to the length of predictions or true labels
element_numbers = list(range(len(y_test_pred)))

##########################################################################################################################
#### Plots for visualization #############################################################################################

'''Below plots are made to visualize what the Random classifier has done and how it has performed'''

# Plot for y_pred
plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot number 1
plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
plt.plot(element_numbers, y_test, label='True Labels', color='black')
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels - {subject}')
plt.legend()


# plt.subplot(1, 4, 2)  # 1 row, 2 columns, plot number 2
# plt.plot(element_numbers, y_test, label='True Labels', color='green')
# plt.xlabel('Element Numbers')
# plt.ylabel('True Labels')
# plt.title(f'True Labels - {subject}')
# plt.legend()

# plt.subplot(2, 4, 3)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{subject}_Test']['hand_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'hand_IMU - {subject}')

# plt.subplot(2, 4, 5)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{subject}_Test']['lowerarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'lowerarm_IMU - {subject}')

# plt.subplot(2, 4, 6)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{subject}_Test']['upperarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'upperarm_IMU - {subject}')

# plt.subplot(2, 4, 7)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{subject}_Test']['shoulder_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'shoulder_IMU - {subject}')

# plt.subplot(2, 4, 8)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{subject}_Test']['sternum_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'sternum_IMU - {subject}')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# plt.figure(figsize=(12, 6))

# plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels vs acceleration data - {subject}')
# plt.legend()
# plt.show()
# # Get feature importances
# importances = clf.feature_importances_


# # Sort feature importances in descending order
# indices = np.argsort(importances)[::-1]

# # Compute confusion matrix for test data
# conf_matrix = confusion_matrix(y_test, y_test_pred)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix for Test Data')
# plt.show()

# # Plot the feature importances
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Importance")
# plt.show()

# # Visualize one of the decision trees in the Random Forest
# plt.figure(figsize=(150, 10))
# plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
# plt.show()



