import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import subprocess

subprocess.run(['python', 'Mean_code_v1.py'])


### Beschrijving: RMS-model that calculates the RMS-value for every row working down. 


# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

x_acceleration2 = acc['drinking_HealthySubject2_Test']['hand_IMU']

def RMS(data):
    #square all separate values in the dataset
    z = np.square(data)

    #count the number of rows in the dataset
    length = len(data[0])
    print(length)
    
    #calculate RMS
    values = np.sqrt(np.sum(z, axis = 1)/length)
    

    #add to original dataset
    rms = np.hstack((data, np.expand_dims(values, axis=1)))

    return values, rms

print('original data: ', x_acceleration2)
print('new dataset: ', RMS(x_acceleration2)[0])

Trans = x_acceleration2[0].T




plt.figure()
plt.plot(RMS(x_acceleration2)[1])
plt.show()