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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
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
subject = 2
subjects_train = [f'drinking_HealthySubject{subject}_Test']
subjects_test = [f'drinking_HealthySubject{subject}_Test']

label = all_labels[subject - 2]
train_labels = label[:1524]
test_labels = label[1524:]
print(test_labels)
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
    labels_train.append(item[1])

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
        # ,acc_entropy_imu,rot_entropy_imu
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_train.append(np.hstack(combined_data_patient))

'''Arrays for all combined train data'''
combined_X_data_train = np.concatenate(X_data_patients_train)
X_train = combined_X_data_train[:1524]
print(X_train.shape)
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
        combined_data_patient.append(combined_data_imu)  # Append each sensor's data

    # Stack the data from all sensors for this patient
    X_data_patients_test.append(np.hstack(combined_data_patient))

'''Combine data from all patients'''
combined_X_data_test = np.concatenate(X_data_patients_test)
X_test = combined_X_data_test[1524:]

print(X_test.shape) ##test print to see the general shape

########################################################################################################################
################ RANDOM FOREST CLASSIFIER ##############################################################################

'''  Below all parameters are set up to run the random forest classifier '''

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
print("y_test_pred",len(y_test_pred))
y_train_pred = clf.predict(X_train)

print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))


importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(X_train.shape[1]), indices)
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Importance")
# plt.show()

num_classes = len(np.unique(y_train))
n_components_lda = min(num_classes - 1, X_train.shape[1])

lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf_lda = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf_lda.fit(X_train_lda, y_train)
y_test_pred_lda = clf_lda.predict(X_test_lda)

clf_pca = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_test_pred_pca = clf_pca.predict(X_test_pca)

print("Classification Report of test data for LDA:")
print(classification_report(y_test, y_test_pred_lda))

print("Classification Report of test data for PCA:")
print(classification_report(y_test, y_test_pred_pca, zero_division=1))

lda_feature_importance = np.abs(lda.coef_[0])

n_features_lda = lda.n_features_in_

lda_feature_importance /= np.sum(lda_feature_importance)

#Get the indices of the most important features
important_features_indices = np.argsort(lda_feature_importance)[::-1]

# Print the most important features
top_n = 30  # Number of top features to print
#print(f"Top {top_n} most important features from LDA:")
#for i in range(top_n):
    #print(f"Feature {important_features_indices[i]}: Importance {lda_feature_importance[important_features_indices[i]]:.4f}")

# print("Feature Importances from LDA:")
# print(lda_feature_importance)

pca_explained_variance_ratio = pca.explained_variance_ratio_

# print("Explained Variance Ratios from PCA:")
# print(pca_explained_variance_ratio)

pca_feature_importance = np.cumsum(pca_explained_variance_ratio)

pca_feature_importance /= np.sum(pca_feature_importance)

# print("Feature Importances from PCA:")
# print(pca_feature_importance)

# plt.figure(figsize=(10, 6))
# plt.bar(range(n_features_lda), lda_feature_importance, align="center", color='orange', label='LDA')
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Importance (LDA)")
# plt.legend()

# plt.figure(figsize=(10, 6))
# plt.bar(range(X_train_pca.shape[1]), pca_feature_importance, align="center", color='green', label='PCA')
# plt.xlabel("PCA Component Index")
# plt.ylabel("Feature Importance (PCA)")
# plt.legend()
# plt.show()
