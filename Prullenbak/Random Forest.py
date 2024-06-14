import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree


# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
#pre = np.load("data_Preprocessed.npy", allow_pickle=True).item()


acc_data2 = acc['drinking_HealthySubject6_Test']
rot_data2 = rot['drinking_HealthySubject6_Test']


annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
print(annotation2)

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}

# Map the letters to numbers in the loaded array
mapped_labels = [[item[0], item[1], label_mapping[item[2]]] for item in annotation2]

# Convert the mapped labels list to a NumPy array
annotation2_numbers = np.array(mapped_labels)

#print(annotation2_numbers)

x_acceleration = acc['drinking_HealthySubject6_Test']['hand_IMU']
Hz = len(x_acceleration)/38.1
print(Hz)

# for subject in subjects:
#     # Extract acceleration data for the current subject and IMU location
#     acc_data = acc[subject]['hand_IMU']
#     # Extract rotation data for the current subject and IMU location
#     rot_data = rot[subject]['hand_IMU']

#     # Extract X, Y, and Z acceleration
#     x_acceleration = acc_data[:, 0]
#     y_acceleration = acc_data[:, 1]
#     z_acceleration = acc_data[:, 2]

#     # Extract X, Y, and Z rotation
#     x_rotation = rot_data[:, 0]
#     y_rotation = rot_data[:, 1]
#     z_rotation = rot_data[:, 2]

#     # Plot acceleration data
#     plt.figure(figsize=(14, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(x_acceleration, label='X Acceleration')
#     plt.plot(y_acceleration, label='Y Acceleration')
#     plt.plot(z_acceleration, label='Z Acceleration')
#     plt.title(f'Acceleration Data for {subject} - hand_IMU')
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration')
#     plt.legend()
#     plt.grid(True)

#     # Plot rotation data
#     plt.subplot(1, 2, 2)
#     plt.plot(x_rotation, label='X Rotation')
#     plt.plot(y_rotation, label='Y Rotation')
#     plt.plot(z_rotation, label='Z Rotation')
#     plt.title(f'Rotation Data for {subject} - hand_IMU')
#     plt.xlabel('Time')
#     plt.ylabel('Rotation')
#     plt.legend()
# #     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

X_data_per_sensor = []

for imu_location in imu_locations:
    acc_data_imu = acc_data2[imu_location]
    rot_data_imu = rot_data2[imu_location]
    
    # Combine accelerometer and gyroscope data horizontally
    combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
    
    X_data_per_sensor.extend(combined_data_imu.T)

# Combine data from the selected IMU sensors horizontally
X_data = np.vstack(X_data_per_sensor).T
#print(X_data)

# Now X_data contains the combined data from the selected IMU sensors
print(X_data.shape)  # Verify the shape of the combined data

labels_per_measurement = []

for row in annotation2:
    label = label_mapping[row[2]]
    start_time = float(row[0])
    end_time = float(row[1])
    duration = end_time - start_time
    num_measurements = round(duration * Hz)
    print("variables",start_time,end_time,label,duration,num_measurements)
    labels_per_measurement.extend([label] * num_measurements)

#print(labels_per_measurement)
print(len(labels_per_measurement))

# Load X_data and labels_per_measurement
# Assuming X_data and labels_per_measurement are already defined

# Split the data into training and testing sets
#deze weggehaalt bij multiple random people forest omdat we volledige data van patient willen halen v
X_train, X_test, y_train, y_test = train_test_split(X_data, labels_per_measurement, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

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
plt.figure(figsize=(70, 10))
plot_tree(clf.estimators_[0], feature_names=[f'feature {i}' for i in range(X_train.shape[1])], filled=True)
plt.show()

# # Sample data point (measurement frame) for prediction
# sample_data_point = X_test[0]  # You can replace this with any other data point

# # Make prediction for the sample data point
# predicted_label = clf.predict([sample_data_point])[0]

# # Map the predicted label back to the original movement type
# predicted_movement = {v: k for k, v in label_mapping.items()}[predicted_label]

# # Print the predicted movement type
# print("Predicted Movement Type:", predicted_movement)