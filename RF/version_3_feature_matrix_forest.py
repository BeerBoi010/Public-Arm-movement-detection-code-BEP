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
sampling_window_entropy = 3
test_person = 7
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices


subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
        'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']
print(subjects_test)

test_labels = all_labels[test_person - 2]
#print("test labels:",test_labels)

all_labels.pop(test_person - 2)
train_labels = all_labels
#print("train labels:",train_labels)

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
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data
        print(combined_data_imu.shape)
    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

'''Arrays for all combined train data'''
combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train
print(combined_X_data_train.shape)
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



        combined_data_imu = np.hstack((acc_rms_imu, rot_rms_imu, acc_mean_imu, rot_mean_imu,acc_slope_imu,rot_slope_imu,
                                       acc_max_imu,rot_max_imu,acc_min_imu,rot_min_imu,acc_STD_imu,rot_STD_imu))
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_test.append(np.hstack(combined_data_patient))

'''Combine data from all patients'''
combined_X_data_test = np.concatenate(X_data_patients_test)
X_test = combined_X_data_test

print(combined_X_data_test.shape) ##test print to see the general shape

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
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 4, 1)  # 1 row, 2 columns, plot number 1
# plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels - {subject}')
# plt.legend()


# plt.subplot(2, 4, 2)  # 1 row, 2 columns, plot number 2
# plt.plot(element_numbers, y_test, label='True Labels', color='green')
# plt.xlabel('Element Numbers')
# plt.ylabel('True Labels')
# plt.title(f'True Labels - {subject}')
# plt.legend()

# plt.subplot(2, 4, 3)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'hand_IMU - {subject}')

# plt.subplot(2, 4, 5)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['lowerarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'lowerarm_IMU - {subject}')

# plt.subplot(2, 4, 6)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['upperarm_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'upperarm_IMU - {subject}')

# plt.subplot(2, 4, 7)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['shoulder_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'shoulder_IMU - {subject}')

# plt.subplot(2, 4, 8)  # 1 row, 2 columns, plot number 3
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['sternum_IMU'])
# plt.xlabel('Element number')
# plt.ylabel('acceleration value')
# plt.title(f'sternum_IMU - {subject}')

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels vs acceleration data - {subject}')
# plt.legend()
# plt.show()

# Get feature importances
importances = clf.feature_importances_


# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

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
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()

# # Visualize one of the decision trees in the Random Forest
# plt.figure(figsize=(150, 10))
# plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
# plt.show()



