import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

test_person = 5
bin_size = 5
bin_val = int(1905/bin_size)

print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

##################################################################################################################################
Y_test_labels = all_labels[test_person - 2]
all_labels.pop(test_person - 2)
Y_train_labels = all_labels

# labels_train = []
# ###### for-loops to make annotation list for random forest method ###########################################################################
# # Iterate over each item in Y_train_labels
# for item in Y_train_labels:
#     # Iterate over the item, taking every 5th element
#     for i in range(0, len(item), 5):
#         labels_train.append(item[i][1])  # Append the 2nd element of every 5th sublist
# print("labels train", labels_train, len(labels_train))

# labels_test = []

# for i in range(0, len(Y_test_labels), 5):
#         labels_test.append(Y_test_labels[i][1]) 
# print("labels test", labels_test, len(labels_test))

# # Dictionary to map labels to numerical values
# label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# # Convert labels to numerical values
# y_train = [label_mapping[label] for label in labels_train]
# y_test = [label_mapping[label] for label in labels_test]



#############################################################################################################################
#stacking of acceleration and rotation matrices for all imu sensors next to each other. This way we can prime the data before the feature extraction.
X_data_patients_dict = {}

for subject in subjects_train:
    combined_data_patient = []
    for imu in acc[subject]:
        combined_data_imu = np.hstack((acc[subject][imu], rot[subject][imu]))
        combined_data_patient.append(combined_data_imu)
    #Dictionary with the combined acc and rot data per subject
    X_data_patients_dict[subject] = np.hstack((combined_data_patient))

# Combine data for all subjects into a single array
combined_X_data_train = np.concatenate(list(X_data_patients_dict.values()))
FullCombinedData = combined_X_data_train


################## Setting up the feature matrix ###################
feature_dict = {'drinking_HealthySubject2_Test': [],'drinking_HealthySubject3_Test': [], 'drinking_HealthySubject4_Test': [],
                  
                        'drinking_HealthySubject6_Test': [],'drinking_HealthySubject7_Test': []}
for patient in X_data_patients_dict:
    #Calls the array for one subject and splits it in equal parts of five
    X_data_patients_dict[patient] = np.array_split(X_data_patients_dict[patient], bin_val)

    for split in X_data_patients_dict[patient]:

        #Setting up features that loop through the columns: mean_x_acc,mean_y_acc....,Mean_x_rot. For all featured and 5 imu sensors so
        #a row of 5*5*6 = 150 features 
        Mean = np.mean(split, axis=0)
        STD = np.std(split, axis=0)
        RMS = np.sqrt(np.mean(split**2, axis=0))  # RMS value of each column
        MIN = np.min(split, axis=0)
        MAX = np.max(split, axis=0)
        #appends all features in a dictionary for each patient 
        feature_dict[patient].append(np.hstack((Mean,STD,RMS,MIN,MAX)))

# feature_matrix['drinking_HealthySubject2_Test'] = np.array(feature_matrix['drinking_HealthySubject2_Test'])
# print(feature_matrix['drinking_HealthySubject2_Test'],feature_matrix['drinking_HealthySubject2_Test'].shape)
# #print(X_data_patients_dict['drinking_HealthySubject2_Test'],X_data_patients_dict['drinking_HealthySubject2_Test'].shape

# Combine all feature arrays into a single array
compressed_array_train = np.concatenate(list(feature_dict.values()), axis=0)


###############################################test################################

#stacking of acceleration and rotation matrices for all imu sensors next to each other. This way we can prime the data before the feature extraction.
X_data_patients_dict = {}

for subject in subjects_test:
    print(subject)
    combined_data_patient = []
    for imu in acc[subject]:
        combined_data_imu = np.hstack((acc[subject][imu], rot[subject][imu]))
        combined_data_patient.append(combined_data_imu)
    #Dictionary with the combined acc and rot data per subject
    X_data_patients_dict[subject] = np.hstack((combined_data_patient))

# Combine data for all subjects into a single array
combined_X_data_train = np.concatenate(list(X_data_patients_dict.values()))
FullCombinedData = combined_X_data_train


################## Setting up the feature matrix ###################
feature_dict = {'drinking_HealthySubject5_Test': []}
for patient in X_data_patients_dict:
    #Calls the array for one subject and splits it in equal parts of five
    X_data_patients_dict[patient] = np.array_split(X_data_patients_dict[patient], bin_val)

    for split in X_data_patients_dict[patient]:

        #Setting up features that loop through the columns: mean_x_acc,mean_y_acc....,Mean_x_rot. For all featured and 5 imu sensors so
        #a row of 5*5*6 = 150 features 
        Mean = np.mean(split, axis=0)
        STD = np.std(split, axis=0)
        RMS = np.sqrt(np.mean(split**2, axis=0))  # RMS value of each column
        MIN = np.min(split, axis=0)
        MAX = np.max(split, axis=0)
        #appends all features in a dictionary for each patient 
        feature_dict[patient].append(np.hstack((Mean,STD,RMS,MIN,MAX)))

# feature_matrix['drinking_HealthySubject2_Test'] = np.array(feature_matrix['drinking_HealthySubject2_Test'])
# print(feature_matrix['drinking_HealthySubject2_Test'],feature_matrix['drinking_HealthySubject2_Test'].shape)
# #print(X_data_patients_dict['drinking_HealthySubject2_Test'],X_data_patients_dict['drinking_HealthySubject2_Test'].shape

# Combine all feature arrays into a single array
compressed_array_test = np.concatenate(list(feature_dict.values()), axis=0)


print(compressed_array_test.shape,compressed_array_train.shape)

clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=1,max_depth=10, random_state=42)
clf.fit(compressed_array_train, y_train)

y_test_pred = clf.predict(compressed_array_test)
y_train_pred = clf.predict(compressed_array_train)


print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))

element_numbers = list(range(len(y_test_pred)))

### Setting up plots to illustrate code
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(element_numbers, y_test_pred, label='Predictions', color='blue')
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels - {subjects_test[0]}')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(element_numbers, y_test, label='True Labels', color='green')
plt.xlabel('Element Numbers')
plt.ylabel('True Labels')
plt.title(f'True Labels - {subjects_test[0]}')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'hand_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 5)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['lowerarm_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'lowerarm_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 6)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['upperarm_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'upperarm_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 7)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['shoulder_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'shoulder_IMU - {subjects_test[0]}')

plt.subplot(2, 4, 8)
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['sternum_IMU'])
plt.xlabel('Element number')
plt.ylabel('Acceleration value')
plt.title(f'sternum_IMU - {subjects_test[0]}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(element_numbers, y_test_pred, label='Predictions', color='black')
plt.plot(acc[f'drinking_HealthySubject{test_person}_Test']['hand_IMU'])
plt.xlabel('Element Numbers')
plt.ylabel('Predicted Labels')
plt.title(f'Predicted Labels vs Acceleration Data - {subjects_test[0]}')
plt.legend()
plt.show()

# Compute confusion matrix for test data
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Label maps for confusion matrix
label_mapping = {0: 'N', 1: 'A', 2: 'B', 3: 'C'}

# Plot confusion matrix
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_mapping[key] for key in label_mapping.keys()],
            yticklabels=[label_mapping[key] for key in label_mapping.keys()])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix of drinking_HealthySubject{test_person}_Test')
plt.show()

importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(compressed_array_train.shape[1]), importances[indices], align="center")
plt.xticks(range(compressed_array_train.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.show()

num_classes = len(np.unique(y_train))
n_components_lda = min(num_classes - 1, compressed_array_train.shape[1])

lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda.fit_transform(compressed_array_train, y_train)
X_test_lda = lda.transform(compressed_array_test)

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(compressed_array_train)
X_test_pca = pca.transform(compressed_array_test)

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
print(f"Top {top_n} most important features from LDA:")
for i in range(top_n):
    print(f"Feature {important_features_indices[i]}: Importance {lda_feature_importance[important_features_indices[i]]:.4f}")

print("Feature Importances from LDA:")
print(lda_feature_importance)

pca_explained_variance_ratio = pca.explained_variance_ratio_

print("Explained Variance Ratios from PCA:")
print(pca_explained_variance_ratio)

pca_feature_importance = np.cumsum(pca_explained_variance_ratio)

pca_feature_importance /= np.sum(pca_feature_importance)

print("Feature Importances from PCA:")
print(pca_feature_importance)

plt.figure(figsize=(10, 6))
plt.bar(range(n_features_lda), lda_feature_importance, align="center", color='orange', label='LDA')
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance (LDA)")
plt.legend()

plt.figure(figsize=(10, 6))
plt.bar(range(X_train_pca.shape[1]), pca_feature_importance, align="center", color='green', label='PCA')
plt.xlabel("PCA Component Index")
plt.ylabel("Feature Importance (PCA)")
plt.legend()
plt.show()
