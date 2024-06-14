import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###Description: Calculates mean-value for given data over a sampling window, working down and giving values for everor of the dataset.


# #variables
# sampling_window = 3
# min_periods = 1

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

#######################################################

############train####################

def Mean_train(train_amount,sampling_window,min_periods):
    ###function: calculate mean-values for all patients, with acc and gyr data.
    mean_data_all_patients = {}

    # Iterate over each patient
    for subject in train_amount:

        #calcluation of values for every imu sensor
        mean_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        
        combined_data_patient = []
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)


            #The rolling mean calculates the rolling mean for the entire row
            mean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            mean_rot= dataset_rot.rolling(sampling_window, min_periods).mean()


            # Store RMS data for the current sensor location in the dictionary
            mean_data_patient[imu_location] = {'acc_mean': mean_acc, 'rot_mean': mean_rot}
        
        # Store RMS data for the current patient in the dictionary
        mean_data_all_patients[subject] = mean_data_patient
    
    # Return the dictionary containing RMS data for all patients
    return mean_data_all_patients

#############################################################################

##########test############
def Mean_test(test_amount,sampling_window,min_periods):
    ###function: calculate mean-values for all patients, with acc and gyr data.
    mean_data_all_patients = {}

    # Iterate over each patient
    for subject in test_amount:

        #calcluation of values for every imu sensor
        mean_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)


            #The rolling mean calculates the rolling mean for the entire row
            mean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            mean_rot= dataset_rot.rolling(sampling_window, min_periods).mean()


            # Store mean data for the current sensor location in the dictionary
            mean_data_patient[imu_location] = {'acc_mean': mean_acc, 'rot_mean': mean_rot}
        
        # Store mean data for the current patient in the dictionary
        mean_data_all_patients[subject] = mean_data_patient
    
    # Return the dictionary containing mean data for all patients
    return mean_data_all_patients

#print(Mean_train())


