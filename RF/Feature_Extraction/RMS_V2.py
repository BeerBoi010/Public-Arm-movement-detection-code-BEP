import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###Description: Calculates RMS-value for given data over a sampling window, working down and giving values for everor of the dataset.


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


##################################################################################################

####train##################

def RMS_train(train_amount,sampling_window, min_periods):
    ###function: calculate RMS-values for all patients, with acc and gyr data.
    rms_data_all_patients = {}

    # Iterate over each patient
    for subject in train_amount:

        #calcluation of values for every imu sensor
        rms_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            
            #calcluation of Squared matrices for one sensor, for one patient
            Squared_acc = np.square(acc_data_imu)
            Squared_rot = np.square(rot_data_imu)

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(Squared_acc)
            dataset_rot = pd.DataFrame(Squared_rot)

            #The rolling mean calculates the rolling mean for the entire row
            Squaredmean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            Squaredmean_rot = dataset_rot.rolling(sampling_window, min_periods).mean()

            RMS_acc = np.sqrt(Squaredmean_acc)
            RMS_rot = np.sqrt(Squaredmean_rot)

            # Store RMS data for the current sensor location in the dictionary
            rms_data_patient[imu_location] = {'acc_rms': RMS_acc, 'rot_rms': RMS_rot}
        
        # Store RMS data for the current patient in the dictionary
        rms_data_all_patients[subject] = rms_data_patient
    
    # Return the dictionary containing RMS data for all patients
    return rms_data_all_patients


############################################################################################ 

####test#####

def RMS_test(test_amount, sampling_window, min_periods):
    ###function: calculate RMS-values for all patients, with acc and gyr data.
    rms_data_all_patients = {}

    # Iterate over each patient
    for subject in test_amount:

        #calcluation of values for every imu sensor
        rms_data_patient = {}
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        

        # Combine accelerometer and gyroscope data horizontally
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            
            #calcluation of Squared matrices for one sensor, for one patient
            Squared_acc = np.square(acc_data_imu)
            Squared_rot = np.square(rot_data_imu)

            #open up a pandas to add a rolling mean for calculations
            dataset_acc = pd.DataFrame(Squared_acc)
            dataset_rot = pd.DataFrame(Squared_rot)

            #The rolling mean calculates the rolling mean for the entire row
            Squaredmean_acc= dataset_acc.rolling(sampling_window, min_periods).mean()
            Squaredmean_rot = dataset_rot.rolling(sampling_window, min_periods).mean()

            RMS_acc = np.sqrt(Squaredmean_acc)
            RMS_rot = np.sqrt(Squaredmean_rot)

            # Store RMS data for the current sensor location in the dictionary
            rms_data_patient[imu_location] = {'acc_rms': RMS_acc, 'rot_rms': RMS_rot}
        
        # Store RMS data for the current patient in the dictionary
        rms_data_all_patients[subject] = rms_data_patient
    
    # Return the dictionary containing RMS data for all patients
    return rms_data_all_patients

#print(RMS_train(5,3,1))
#print(RMS_train())



