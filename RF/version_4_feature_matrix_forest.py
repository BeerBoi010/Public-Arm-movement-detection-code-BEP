#### description:changing data so you dont input the whole dataset of a patient but their trial indivually/ testing for different window sizes

### Importing of necessary libraries ###############################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation


##### VARIABLES ######################################################################################################
# '''later toevoegen dat random wordt gekozen wie train en test is'''

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

# ########must be  into movements(not per measurement!)######################

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']
#print(subjects_test)

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

Y_train_labels = train_labels
Y_test_labels = test_labels


# print(X_test_Slope["drinking_HealthySubject2_Test"]['hand_IMU']["acc_slope"][0])
# corr,_ = pearsonr(X_test_Slope["drinking_HealthySubject2_Test"]['hand_IMU']["acc_slope"][0],X_test_Max["drinking_HealthySubject2_Test"]['hand_IMU']["acc_max"][0])

# print("correlation between slope and max is:",corr)


############################must be labels per movement#########################3
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
#print(combined_X_data_train.shape)
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

#print(combined_X_data_test.shape) ##test print to see the general shape

########################################################################################################################
################ RANDOM FOREST CLASSIFIER ##############################################################################

'''  Below all parameters are set up to run the random forest classifier '''

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
#print("y_test_pred",len(y_test_pred))
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
# #plt.show()

# plt.figure(figsize=(12, 6))

# plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
# plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
# plt.xlabel('Element Numbers')
# plt.ylabel('Predicted Labels')
# plt.title(f'Predicted Labels vs acceleration data - {subject}')
# plt.legend()
# #plt.show()
# Get feature importances
importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances For Grid search Subject 7")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()

# Calculate the number of unique classes in y_train
num_classes = len(np.unique(y_train))

# Set n_components for LDA
n_components_lda = min(num_classes - 1, X_train.shape[1])  # Ensure n_components is <= min(num_classes - 1, num_features)

# Fit LDA
lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Fit PCA
pca = PCA(n_components=None)  # Set n_components=None to keep all components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Random Forest classifier on LDA-transformed data
clf_lda = RandomForestClassifier(n_estimators=100, random_state=42)
clf_lda.fit(X_train_lda, y_train)
y_test_pred_lda = clf_lda.predict(X_test_lda)

# Train Random Forest classifier on PCA-transformed data
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_test_pred_pca = clf_pca.predict(X_test_pca)

# Display classification report of test data for LDA
print("Classification Report of test data for LDA:")
print(classification_report(y_test, y_test_pred_lda))

# Display classification report of test data for PCA with zero_division parameter set
print("Classification Report of test data for PCA:")
print(classification_report(y_test, y_test_pred_pca, zero_division=1))


# Get feature importances from LDA and PCA
lda_feature_importance = np.abs(lda.coef_[0])  # Importance of features in LDA space

# Get the number of input features
n_features_lda = lda.n_features_in_

# Get feature importances from LDA
lda_feature_importance = np.abs(lda.coef_[0])  # Importance of features in LDA space

# Optionally, you can normalize the feature importances
lda_feature_importance /= np.sum(lda_feature_importance)  # Normalize to sum up to 1 if needed

# Display the feature importances
# print("Feature Importances from LDA:")
# print(lda_feature_importance)

# Get explained variance ratios from PCA
pca_explained_variance_ratio = pca.explained_variance_ratio_

# Display the explained variance ratios
# print("Explained Variance Ratios from PCA:")
# print(pca_explained_variance_ratio)

# Compute feature importances from explained variance ratios
pca_feature_importance = np.cumsum(pca_explained_variance_ratio)

# Optionally, you can normalize the feature importances
pca_feature_importance /= np.sum(pca_feature_importance)  # Normalize to sum up to 1 if needed

# # Display the feature importances
# print("Feature Importances from PCA:")
# print(pca_feature_importance)


# Plot the feature importances obtained from LDA
plt.figure(figsize=(10, 6))
plt.bar(range(n_features_lda), lda_feature_importance, align="center", color='orange', label='LDA')
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance (LDA)")
plt.legend()

# Plot the feature importances obtained from PCA
plt.figure(figsize=(10, 6))
plt.bar(range(X_train_pca.shape[1]), pca_feature_importance, align="center", color='green', label='PCA')
plt.xlabel("PCA Component Index")
plt.ylabel("Feature Importance (PCA)")
plt.legend()
plt.show()



