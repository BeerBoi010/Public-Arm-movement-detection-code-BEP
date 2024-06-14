import numpy as np
import pandas as pd

imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']


subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']


acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

def RMS_all_subjects(sampling_window, min_periods):
    rms_data_all_patients = {}

    for subject in subjects:
        rms_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            Squared_acc = np.square(acc_data_imu)
            Squared_rot = np.square(rot_data_imu)

            dataset_acc = pd.DataFrame(Squared_acc)
            dataset_rot = pd.DataFrame(Squared_rot)

            Squaredmean_acc = dataset_acc.rolling(sampling_window, min_periods).mean()
            Squaredmean_rot = dataset_rot.rolling(sampling_window, min_periods).mean()

            RMS_acc = np.sqrt(Squaredmean_acc)
            RMS_rot = np.sqrt(Squaredmean_rot)

            rms_data_patient[imu_location] = {'acc_rms': RMS_acc, 'rot_rms': RMS_rot}

        rms_data_all_patients[subject] = rms_data_patient
    
    return rms_data_all_patients


sampling_window = 3
min_periods = 1
all_rms_data = RMS_all_subjects(sampling_window, min_periods)
print(all_rms_data)