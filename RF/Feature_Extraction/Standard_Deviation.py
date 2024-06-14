import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### VARIABLES ###############################################################  
degree = 1
STD_samplingwindow = 3

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

#### Function to compute the rolling standard deviation  ######################################################
def compute_STD(data, sampling_window,min_periods):
    '''code to calculate the standard deviation of a certain window size usng rolling.std'''
    STD = data.rolling(sampling_window,min_periods = min_periods).std()
    return STD

##############################################################################################################3

# Modify your existing code to use compute_STD function
def STD_train(train_amount, sampling_window, min_periods):
    STD_data_all_patients = {}

    for subject in train_amount:
        STD_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            STD_acc = compute_STD(dataset_acc, sampling_window, min_periods)
            STD_rot = compute_STD(dataset_rot, sampling_window, min_periods)
            STD_acc.bfill(inplace=True) # Fill NaN with the next valid value
            STD_rot.bfill(inplace=True) # Fill NaN with the next valid value

            STD_data_patient[imu_location] = {'acc_STD': STD_acc, 'rot_STD': STD_rot}

        STD_data_all_patients[subject] = STD_data_patient

    return STD_data_all_patients

def STD_test(test_amount, sampling_window, min_periods):
    STD_data_all_patients = {}

    for subject in test_amount:
        STD_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            STD_acc = compute_STD(dataset_acc, sampling_window, min_periods)
            STD_rot = compute_STD(dataset_rot, sampling_window, min_periods)
            STD_acc.bfill(inplace=True) # Fill NaN with the next valid value
            STD_rot.bfill(inplace=True) # Fill NaN with the next valid value
            

            STD_data_patient[imu_location] = {'acc_STD': STD_acc, 'rot_STD': STD_rot}

        STD_data_all_patients[subject] = STD_data_patient

    return STD_data_all_patients

# Function to plot the STD data
def plot_STD_data(STD_data):
    for subject, imu_data in STD_data.items():
        for imu_location, STD_values in imu_data.items():
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(STD_values['acc_STD'], label='Acc STD')
            plt.title(f"STD Data for {subject} - {imu_location} (Accelerometer)")
            plt.xlabel("Sample")
            plt.ylabel("STD")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(STD_values['rot_STD'], label='Rot STD')
            plt.title(f"STD Data for {subject} - {imu_location} (Gyroscope)")
            plt.xlabel("Sample")
            plt.ylabel("STD")
            plt.legend()

            plt.tight_layout()
            plt.show()

#print(STD_test(5,3,1))
