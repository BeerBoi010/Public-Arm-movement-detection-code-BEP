import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

acc_data = acc['drinking_HealthySubject2_Test']['hand_IMU']
x_acceleration = acc_data[:, 0]

Hz = len(x_acceleration)/38.1

#print(Hz)

###Panda data editing ###

annotations = np.load("Data_tests/time_ranges_subject_2.npy", allow_pickle=True)
df = pd.DataFrame(annotations)
labels = {'N': 0, 'A': 1, 'B' : 2, 'C':3}
anntest = df.replace(labels)


print('replaced dataset is:', anntest)
df.head()
plt.figure(figsize=(8, 6))
plt.table(cellText=df.values, colLabels=df.columns, loc='center')
plt.axis('off')  # Turn off the axes
plt.show()

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',
                 
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']
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
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()

# X_subject_2 = []
# y_subject_2 = []

# for start, end, label in annotation2:
#     for imu_location in imu_locations:
#         acc_data = acc[f'drinking_HealthySubject2_Test'][imu_location]
#         rot_data = rot[f'drinking_HealthySubject2_Test'][imu_location]
#         combined_data = np.concatenate((acc_data, rot_data), axis=1)
#         X_subject_2.append(combined_data)
#         y_subject_2.append(label)

# X_subject_2 = np.concatenate(X_subject_2)
# y_subject_2 = np.array(y_subject_2)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_subject_2, y_subject_2, test_size=0.2, random_state=42)