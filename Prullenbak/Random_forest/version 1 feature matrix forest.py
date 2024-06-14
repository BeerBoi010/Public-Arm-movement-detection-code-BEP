import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import sys


###Description: Start of exploring acc, rot, mean. Found out taking those values of entire measurement series does not impact 
###accuracy of the model. 

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
print(len(acc['drinking_HealthySubject6_Test']['hand_IMU']))
# Create lists to store data and labels for each patient
X_data_patients_train = []
labels_patients_train = []

# Iterate over each patient
for subject in subjects[:4]:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 


    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        #print("variables",start_time,end_time,label,duration,num_measurements)
        labels_patient.extend([label] * num_measurements)
    
    if subject == 'drinking_HealthySubject6_Test':
        labels_patient = labels_patient[:-5]  # Delete the last 5 labels

    combined_data_patient = []
    for imu_location in imu_locations:

        acc_data_imu = acc_data_patient[imu_location]
        rot_data_imu = rot_data_patient[imu_location]

        # Calculate min, max, and mean values for XYZ acceleration and rotation
        acc_min = np.min(acc_data_imu, axis=0)
        acc_max = np.max(acc_data_imu, axis=0)
        acc_mean = np.mean(acc_data_imu, axis=0)
    
        rot_min = np.min(rot_data_imu, axis=0)
        rot_max = np.max(rot_data_imu, axis=0)
        rot_mean = np.mean(rot_data_imu, axis=0)

        # Expand min, max, and mean values to have the same number of rows as acc_data_imu and rot_data_imu
        num_rows = acc_data_imu.shape[0]
        acc_min = np.tile(acc_min, (num_rows, 1))
        acc_max = np.tile(acc_max, (num_rows, 1))
        acc_mean = np.tile(acc_mean, (num_rows, 1))
        rot_min = np.tile(rot_min, (num_rows, 1))
        rot_max = np.tile(rot_max, (num_rows, 1))
        rot_mean = np.tile(rot_mean, (num_rows, 1))

        combined_data_imu = np.hstack((acc_data_imu, rot_data_imu,acc_min,acc_max,acc_mean,rot_min,rot_max,rot_mean))
        combined_data_patient.extend(combined_data_imu.T)
    
    # Add data and labels to the lists
    X_data_patients_train.append(np.vstack(combined_data_patient).T)
    labels_patients_train.append(labels_patient)

# Combine data and labels from all patients
combined_X_data = np.concatenate(X_data_patients_train)
combined_labels = np.concatenate(labels_patients_train)

#print(combined_labels)
print(combined_X_data.shape,combined_labels.shape)

# Split the combined dataset and label array
#X_train, X_test, y_train, y_test = train_test_split(combined_X_data, combined_labels, test_size=0.2, random_state=42)
X_train = combined_X_data
y_train = combined_labels

subjects_test = subjects[4:]

# Create lists to store data and labels for each patient
X_data_patients_test = []
labels_patients_test = []

# Iterate over each patient
for subject in subjects_test:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = [] 


    for row in annotation[subject]:
        label = int(row[2])
        start_time = float(row[0])
        end_time = float(row[1])
        duration = end_time - start_time
        num_measurements = round(duration * Hz)
        #print("variables",start_time,end_time,label,duration,num_measurements)
        labels_patient.extend([label] * num_measurements)
    
    if subject == 'drinking_HealthySubject6_Test':
        labels_patient = labels_patient[:-5]  # Delete the last 5 labels

    # Combine accelerometer and gyroscope data horizontally
    combined_data_patient = []
    for imu_location in imu_locations:

        acc_data_imu = acc_data_patient[imu_location]
        rot_data_imu = rot_data_patient[imu_location]

        # Calculate min, max, and mean values for XYZ acceleration and rotation
        acc_min = np.min(acc_data_imu, axis=0)
        acc_max = np.max(acc_data_imu, axis=0)
        acc_mean = np.mean(acc_data_imu, axis=0)
    
        rot_min = np.min(rot_data_imu, axis=0)
        rot_max = np.max(rot_data_imu, axis=0)
        rot_mean = np.mean(rot_data_imu, axis=0)

        # Expand min, max, and mean values to have the same number of rows as acc_data_imu and rot_data_imu
        num_rows = acc_data_imu.shape[0]
        acc_min = np.tile(acc_min, (num_rows, 1))
        acc_max = np.tile(acc_max, (num_rows, 1))
        acc_mean = np.tile(acc_mean, (num_rows, 1))
        rot_min = np.tile(rot_min, (num_rows, 1))
        rot_max = np.tile(rot_max, (num_rows, 1))
        rot_mean = np.tile(rot_mean, (num_rows, 1))

        combined_data_imu = np.hstack((acc_data_imu, rot_data_imu,acc_min,acc_max,acc_mean,rot_min,rot_max,rot_mean))
        combined_data_patient.extend(combined_data_imu.T)
    
    # Add data and labels to the lists
    X_data_patients_test.append(np.vstack(combined_data_patient).T)
    labels_patients_test.append(labels_patient)

# Combine data and labels from all patients
combined_X_data = np.concatenate(X_data_patients_test)
combined_labels = np.concatenate(labels_patients_test)

#print(combined_labels)
print(combined_X_data.shape,combined_labels.shape)

X_test = combined_X_data
y_test = combined_labels

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