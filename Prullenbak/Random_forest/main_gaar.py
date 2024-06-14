import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd

####    Importing of necessary functions for algorithm  ###############################################
import RMS_V2
import Mean_V2
import Random_forest.labels_interpolation as labels_interpolation

##### VARIABLES ##########################################################################################
#later toevoegen dat random wordt gekozen wie train en test is 
train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount

### Setting up the test and training sets and labels ############################################################

X_train_RMS = RMS_V2.RMS_train(train_amount, sampling_window, min_periods)

X_test_RMS = RMS_V2.RMS_train(test_amount, sampling_window, min_periods)

X_train_Mean = Mean_V2.Mean_train(train_amount, sampling_window, min_periods)
X_test_Mean = Mean_V2.Mean_test(test_amount, sampling_window, min_periods)
#print(X_test_Mean)
print(X_test_RMS)
Y_train_labels = labels_interpolation.expanded_matrices[:train_amount]
Y_test_labels = labels_interpolation.expanded_matrices[test_amount:]

# Assuming 'rms' and 'mean' are dictionaries containing RMS and mean values respectively
# for each patient and IMU sensor

# Number of patients
num_patients = 6

# Number of IMU sensors per patient
num_sensors = 5

# Number of rows (assuming this is consistent across patients and IMU sensors)
num_rows = len(X_train_RMS['drinking_HealthySubject2_Test'][list(X_train_RMS['drinking_HealthySubject2_Test'].keys())[0]])

# Initialize numpy arrays to store the data
acceleration_rms = np.zeros((num_rows, num_patients * num_sensors * 3))  # 3 axes for acceleration
rotation_rms = np.zeros((num_rows, num_patients * num_sensors * 3))      # 3 axes for rotation
acceleration_mean = np.zeros((num_rows, num_patients * num_sensors * 3)) # 3 axes for acceleration
rotation_mean = np.zeros((num_rows, num_patients * num_sensors * 3))      # 3 axes for rotation

# Iterate over patients
for patient_idx in range(num_patients):
    patient_key = f'drinking_HealthySubject{patient_idx + 2}_Test'  # Assuming patient keys follow this format
    # Iterate over IMU sensors
    for sensor_idx, imu_sensor in enumerate(X_train_RMS[patient_key]):
        # Compute starting column index for the current IMU sensor
        start_col_idx = (patient_idx * num_sensors + sensor_idx) * 6
        # Extract RMS and mean values for the current IMU sensor
        acc_rms = X_train_RMS[patient_key][imu_sensor]['acc_rms']
        rot_rms = X_train_RMS[patient_key][imu_sensor]['rot_rms']
        acc_mean = X_train_Mean[patient_key][imu_sensor]['acc_mean']
        rot_mean = X_train_Mean[patient_key][imu_sensor]['rot_mean']
        # Store values in the corresponding columns
        acceleration_rms[:, start_col_idx:start_col_idx + 3] = acc_rms
        rotation_rms[:, start_col_idx:start_col_idx + 3] = rot_rms
        acceleration_mean[:, start_col_idx:start_col_idx + 3] = acc_mean
        rotation_mean[:, start_col_idx:start_col_idx + 3] = rot_mean

print(acceleration_rms)

# Now, acceleration_rms, rotation_rms, acceleration_mean, and rotation_mean
# contain the organized data as NumPy arrays.













# # Convert your X_train and X_test data to DataFrames
# X_train_RMS_df = pd.DataFrame(X_train_RMS)
# X_test_RMS_df = pd.DataFrame(X_test_RMS)

# X_train_Mean_df = pd.DataFrame(X_train_Mean)
# X_test_Mean_df = pd.DataFrame(X_test_Mean)

# # Horizontally concatenate the data
# X_train_concat = pd.concat([X_train_RMS_df, X_train_Mean_df], axis=1)
# X_test_concat = pd.concat([X_test_RMS_df, X_test_Mean_df], axis=1)


# print(X_train_concat.isnull().sum())
# print(X_train_concat.dtypes)
# print(X_train_concat.head())
# print(X_train_concat.shape)



################ training the model #################################

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

# np.set_printoptions(threshold=sys.maxsize)
# print("True Test labels", y_test,len(y_test))
# print("Predictions Test labels",y_pred_int,len(y_pred_int))

# Splitting y_test_pred and y_test into separate arrays for each patient
# Splitting y_test_pred and y_test into separate arrays for each patient
split_y_pred = np.split(y_test_pred, [1905*i for i in range(1, len(subjects_test)+1)])
split_y_test = np.split(y_test, [1905*i for i in range(1, len(subjects_test)+1)])

#print(split_y_test)
#print(len(split_y_test))

# Iterate over each patient in the test set
for i, subject in enumerate(subjects_test):
    # Extract predictions and true labels for the current patient
    y_pred_patient = split_y_pred[i]
    y_test_patient = split_y_test[i]
    
    # Create an empty list of size equal to the length of predictions or true labels
    element_numbers = list(range(len(y_pred_patient)))

    # Plot for y_pred
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot number 1
    plt.plot(element_numbers, y_pred_patient, label='Predictions', color='blue')
    plt.xlabel('Element Numbers')
    plt.ylabel('Predicted Labels')
    plt.title(f'Predicted Labels - {subject}')
    plt.legend()

    # Plot for y_test
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot number 2
    plt.plot(element_numbers, y_test_patient, label='True Labels', color='green')
    plt.xlabel('Element Numbers')
    plt.ylabel('True Labels')
    plt.title(f'True Labels - {subject}')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Display classification report
print("Classification Report of train data:")
print(classification_report(y_train, y_train_pred))

# Display classification report
print("Classification Report of test data:")
print(classification_report(y_test, y_test_pred))


# Get feature importances
importances = clf.feature_importances_

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


# Visualize one of the decision trees in the Random Forest
plt.figure(figsize=(150, 10))
plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
plt.show()

