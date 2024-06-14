import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys


###Description: Code for calculating the mean over a certain window size. generates a matrix the same size as the input.


#important variables:
sampling_window = 3

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()



#setting up the first testing data
x_acceleration2 = acc['drinking_HealthySubject2_Test']['hand_IMU']
x_accT = x_acceleration2.T

#Setting up the mean
dataset_sub2= pd.DataFrame(x_acceleration2)

#The rolling mean calculates the rolling mean for the entire row
roller= dataset_sub2.rolling(sampling_window, min_periods=3).mean()

#changing the meaned rows to numpy ant transposing them for the plot
x = roller.to_numpy()
mean_acc= x.T

# print(x)
# print(x_accT[0])

plt.figure()
plt.plot(x_accT[0])
# plt.plot(x_plot[0])
plt.show()