import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Subjects for training and testing
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# Function to calculate rolling entropy
def calculate_entropy_rolling(series, window_size):
    rolled = series.rolling(window=window_size, min_periods=1)
    return rolled.apply(lambda x: entropy(np.histogram(x, bins=10, range=(x.min(), x.max()), density=True)[0]), raw=True)

# Function to calculate entropy for training data
def Entropy_train(train_amount, sampling_window):
    entropy_data_all_patients = {}

    for subject in train_amount:
        entropy_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            entropy_acc = dataset_acc.apply(lambda x: calculate_entropy_rolling(x, sampling_window))
            entropy_rot = dataset_rot.apply(lambda x: calculate_entropy_rolling(x, sampling_window))

            entropy_data_patient[imu_location] = {'acc_entropy': entropy_acc, 'rot_entropy': entropy_rot}
        
        entropy_data_all_patients[subject] = entropy_data_patient
    
    return entropy_data_all_patients

# Function to calculate entropy for testing data
def Entropy_test(test_amount, sampling_window):
    entropy_data_all_patients = {}

    for subject in test_amount:
        entropy_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            entropy_acc = dataset_acc.apply(lambda x: calculate_entropy_rolling(x, sampling_window))
            entropy_rot = dataset_rot.apply(lambda x: calculate_entropy_rolling(x, sampling_window))

            entropy_data_patient[imu_location] = {'acc_entropy': entropy_acc, 'rot_entropy': entropy_rot}
        
        entropy_data_all_patients[subject] = entropy_data_patient
    
    return entropy_data_all_patients

# Example usage
sampling_window = 3
train_amount = subjects[:4]  # Example: first 4 subjects for training
test_amount = subjects[4:]   # Example: remaining subjects for testing

# Calculate entropy for training and testing sets
train_entropy_data = Entropy_train(train_amount, sampling_window)
test_entropy_data = Entropy_test(test_amount, sampling_window)

# Print results (or save them as needed)
# print(train_entropy_data)
# print(test_entropy_data)
