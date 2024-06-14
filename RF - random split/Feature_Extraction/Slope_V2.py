import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# Function to compute slope
def compute_slope(data, sampling_window, min_periods):
    slope = data.rolling(window=sampling_window, min_periods=min_periods).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (sampling_window - 1))
    return slope

# Modify your existing code to use compute_slope function
def Slope_train(train_amount, sampling_window, min_periods):
    slope_data_all_patients = {}

    for subject in train_amount:
        slope_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            slope_acc = compute_slope(dataset_acc, sampling_window, min_periods)
            slope_rot = compute_slope(dataset_rot, sampling_window, min_periods)

            slope_data_patient[imu_location] = {'acc_slope': slope_acc, 'rot_slope': slope_rot}

        slope_data_all_patients[subject] = slope_data_patient

    return slope_data_all_patients

def Slope_test(test_amount, sampling_window, min_periods):
    slope_data_all_patients = {}

    for subject in test_amount:
        slope_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            slope_acc = compute_slope(dataset_acc, sampling_window, min_periods)
            slope_rot = compute_slope(dataset_rot, sampling_window, min_periods)

            slope_data_patient[imu_location] = {'acc_slope': slope_acc, 'rot_slope': slope_rot}

        slope_data_all_patients[subject] = slope_data_patient

    return slope_data_all_patients

# Function to plot the slope data
def plot_slope_data(slope_data):
    for subject, imu_data in slope_data.items():
        for imu_location, slope_values in imu_data.items():
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(slope_values['acc_slope'], label='Acc Slope')
            plt.title(f"Slope Data for {subject} - {imu_location} (Accelerometer)")
            plt.xlabel("Sample")
            plt.ylabel("Slope")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(slope_values['rot_slope'], label='Rot Slope')
            plt.title(f"Slope Data for {subject} - {imu_location} (Gyroscope)")
            plt.xlabel("Sample")
            plt.ylabel("Slope")
            plt.legend()

            plt.tight_layout()
            plt.show()

#print(Slope_test(5,3,1))
