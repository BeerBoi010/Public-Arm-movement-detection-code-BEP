import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import sys

###     Beschrijving: Model waartbij de features van de hele data werd gebruikt. Niet nuttig voor onze modellen 
# Define IMU locations
#imu_locations = ['hand_IMU']
#imu_locations = ['hand_IMU', 'lowerarm_IMU']
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']


# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
#pre = np.load("data_Preprocessed.npy", allow_pickle=True).item()


annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation6 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

print(annotation2)

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Map the letters to numbers in the loaded array
mapped_labels2 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation2]
mapped_labels3 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation3]
mapped_labels4 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation4]
mapped_labels5 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation5]
mapped_labels6 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation6]
mapped_labels7 = [[item[0], item[1], label_mapping[item[2]]] for item in annotation7]

# Convert the mapped labels list to a NumPy array
annotation2_numbers = np.array(mapped_labels2)
annotation3_numbers = np.array(mapped_labels3)
annotation4_numbers = np.array(mapped_labels4)
annotation5_numbers = np.array(mapped_labels5)
annotation6_numbers = np.array(mapped_labels6)
annotation7_numbers = np.array(mapped_labels7)

annotation = {'drinking_HealthySubject2_Test':annotation2_numbers,'drinking_HealthySubject3_Test':annotation3_numbers,
              'drinking_HealthySubject4_Test':annotation4_numbers,'drinking_HealthySubject5_Test':annotation5_numbers,
              'drinking_HealthySubject6_Test':annotation6_numbers,'drinking_HealthySubject7_Test':annotation7_numbers
              }

x_acceleration = acc['drinking_HealthySubject2_Test']['hand_IMU']
Hz = len(x_acceleration)/38.1

# Create lists to store data and labels for training and testing
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

# Iterate over each subject
for subject in subjects:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 

    measurement_list = [] 

    # Extract annotations for the current subject
    annotations = annotation[subject]

    # Iterate over each annotation and extract the data
    for row in annotations:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        measurement_list.append(num_measurements)
        labels_patient.append(label)

    # Initialize a list to store data for each movement
    X_data_movements = []

    # Initialize start index
    start_idx = 0

    # Iterate over each annotation and extract the data
    for num_meas in measurement_list:
        acc_data_movement = {imu_location: [] for imu_location in imu_locations}
        rot_data_movement = {imu_location: [] for imu_location in imu_locations}

        # Iterate over each measurement within the movement
        for i in range(start_idx, min(start_idx + num_meas, 1905)):
            for imu_location in imu_locations:
                acc_data_imu = acc_data_patient[imu_location]
                rot_data_imu = rot_data_patient[imu_location]

                # Extract the data for this measurement
                acc_measurement = acc_data_imu[i]
                rot_measurement = rot_data_imu[i]

                acc_data_movement[imu_location].append(acc_measurement)
                rot_data_movement[imu_location].append(rot_measurement)

        # Calculate mean for each IMU sensor
        mean_acc_movement = np.concatenate([np.mean(acc_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
        mean_rot_movement = np.concatenate([np.mean(rot_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
        min_acc_movement = np.concatenate([np.min(acc_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
        max_acc_movement = np.concatenate([np.max(acc_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
        min_rot_movement = np.concatenate([np.min(rot_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])
        max_rot_movement = np.concatenate([np.max(rot_data_movement[imu_loc], axis=0) for imu_loc in imu_locations])

        # Flatten and append the data
        combined_data_movement = np.concatenate([mean_acc_movement, mean_rot_movement, min_acc_movement, max_acc_movement, min_rot_movement, max_rot_movement])
        X_data_movements.append(combined_data_movement)

        # Update the start index for the next movement
        start_idx += num_meas

    # Append the data and labels for the current subject to the appropriate lists
    if subject in subjects[:4]:
        X_train_list.extend(X_data_movements)
        y_train_list.extend(labels_patient)
    else:
        X_test_list.extend(X_data_movements)
        y_test_list.extend(labels_patient)

# Convert lists to numpy arrays
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
print("train",X_train.shape,y_train.shape)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)
print("test",X_test.shape,y_test.shape)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions 
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

subjects_test =  ['drinking_HealthySubject6_Test','drinking_HealthySubject7_Test']

# Splitting y_test_pred and y_test into separate arrays for each patient
split_y_pred = np.split(y_test_pred, [31*i for i in range(1, len(subjects_test)+1)])
split_y_test = np.split(y_test, [31*i for i in range(1, len(subjects_test)+1)])

print(split_y_test)
print(len(split_y_test))

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
    plt.stem(element_numbers, y_pred_patient, label='Predictions')
    plt.xlabel('Element Numbers')
    plt.ylabel('Predicted Labels')
    plt.title(f'Predicted Labels - {subject}')
    plt.legend()

    # Plot for y_test
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot number 2
    plt.stem(element_numbers, y_test_patient, label='True Labels')
    plt.xlabel('Element Numbers')
    plt.ylabel('True Labels')
    plt.title(f'True Labels - {subject}')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


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
plt.figure(figsize=(30, 10))
plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
plt.show()