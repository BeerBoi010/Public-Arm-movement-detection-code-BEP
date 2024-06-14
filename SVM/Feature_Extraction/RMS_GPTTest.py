import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###Description: Calculates RMS-value for given data over a sampling window, working down and giving values for everor of the dataset.

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

sampling_window = 3
min_periods = 1

def compute_rms(data, window, min_periods):
    """ Compute RMS over a sliding window for 2D array """
    data_squared = np.square(data)
    mean_squared = pd.DataFrame(data_squared).rolling(window, min_periods=min_periods).mean()
    return np.sqrt(mean_squared)

def process_imu_data(subjects, imu_locations, acc_data, rot_data, window, min_periods):
    rms_results = {}

    # Calculate RMS for each subject and each IMU sensor
    for subject in subjects:
        rms_results[subject] = {}
        for imu in imu_locations:
            acc_imu_data = acc_data[subject][imu]
            rot_imu_data = rot_data[subject][imu]

            # Calculate RMS for accelerometer and gyroscope data
            rms_acc = compute_rms(acc_imu_data, window, min_periods)
            rms_rot = compute_rms(rot_imu_data, window, min_periods)

            # Store results in a structured format
            rms_results[subject][imu] = {
                'RMS_Accelerometer': rms_acc,
                'RMS_Gyroscope': rms_rot
            }

    return rms_results

# Get a list of subjects and IMU locations from the data dictionaries
subjects = list(acc.keys())
imu_locations = list(acc[subjects[0]].keys())


x = process_imu_data(subjects=subjects,imu_locations=imu_locations,acc_data=acc,rot_data=rot,window=3,min_periods=1)
print(x)