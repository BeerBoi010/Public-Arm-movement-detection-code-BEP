import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']


subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']


acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()


def Mean_all_subjects(sampling_window,min_periods):
    mean_data_all_patients = {}

    for subject in subjects:
        mean_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        
        
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]


            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)


            mean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            mean_rot= dataset_rot.rolling(sampling_window, min_periods).mean()

            mean_data_patient[imu_location] = {'acc_max': mean_acc, 'rot_max': mean_rot}
        

        mean_data_all_patients[subject] = mean_data_patient

    return mean_data_all_patients

sampling_window = 3
min_periods = 1
overall_mean_data = Mean_all_subjects(sampling_window, min_periods)
print(overall_mean_data)


