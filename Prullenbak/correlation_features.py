#### description:changing data so you dont input the whole dataset of a patient but their trial indivually/ testing for different window sizes

### Importing of necessary libraries ###############################################################################################
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr


#### Importing of necessary functions for algorithm  #############################################################################
from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
from Random_forest import labels_interpolation


##### VARIABLES ######################################################################################################
# '''later toevoegen dat random wordt gekozen wie train en test is'''

train_amount = 5
sampling_window = 3
min_periods = 1
test_amount = train_amount
'''' sampling windows with respective values'''
sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 2
#test_person = int(input('Which subject woudl you like to test on (2-7) ? '))

#######################################################################################################################
### Importing and naming of the datasets ##############################################################################

''' Full datasets'''
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices



subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
        'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# ########must be  into movements(not per measurement!)######################

#subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']
#print(subjects_test)

# test_labels = all_labels[test_person - 2]
# #print("test labels:",test_labels)

# all_labels.pop(test_person - 2)
train_labels = all_labels
#print("train labels:",train_labels)

#################################################################################################################
### Setting up the test and training sets with labels ###########################################################

X_train_RMS = RMS_V2.RMS_train(subjects_train, sampling_window_RMS, min_periods)
X_test_RMS = RMS_V2.RMS_test(subjects_test, sampling_window_RMS, min_periods)

X_train_Mean = Mean_V2.Mean_train(subjects_train, sampling_window_mean, min_periods)
X_test_Mean = Mean_V2.Mean_test(subjects_test, sampling_window_mean, min_periods)

X_train_Slope = Slope_V2.Slope_train(subjects_train, sampling_window_slope, min_periods)
X_test_Slope = Slope_V2.Slope_test(subjects_test, sampling_window_slope, min_periods)

X_train_Max = Max_V2.Max_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Max = Max_V2.Max_test(subjects_test, sampling_window_min_max, min_periods)

X_train_Min = Min_V2.Min_train(subjects_train, sampling_window_min_max, min_periods)
X_test_Min = Min_V2.Min_test(subjects_test, sampling_window_min_max, min_periods)

X_train_STD = Standard_Deviation.STD_train(subjects_train, sampling_window_STD, min_periods)
X_test_STD = Standard_Deviation.STD_test(subjects_test, sampling_window_STD, min_periods)


# print(X_test_Slope["drinking_HealthySubject2_Test"]['hand_IMU']["acc_slope"][0])
# corr,_ = pearsonr(X_test_Slope["drinking_HealthySubject2_Test"]['hand_IMU']["acc_slope"][0],X_test_Max["drinking_HealthySubject2_Test"]['hand_IMU']["acc_max"][0])
# print("correlation between slope and max is:",corr)
corr = []
############################### First try with one person ###############################
# for number in range(3):
#     x = X_test_Min["drinking_HealthySubject2_Test"]['hand_IMU']["acc_min"][number]
#     y = X_test_Max["drinking_HealthySubject2_Test"]['hand_IMU']["acc_max"][number]
#     correlation, _= pearsonr(x,y)
#     corr.append(correlation)
#     print(f'the correlation of column {number +1 } is:', {correlation })
########################################################################################3

# Dictionary to hold correlation lists for each subject
correlations = {}

# Iterate over each subject
for subject in subjects:
    # Initialize the correlation list for the current subject
    corr = []
    
    # Get the number of rows (assuming all subjects have the same number of rows)
    num_rows = 3

    #number of imu sensors to iterate over
    

    # Iterate over the rows for the current subject
    for sensor in imu_locations:
        
        for number in range(num_rows):
            x = X_train_Min[subject][sensor]["acc_min"][number]
            y = X_train_Max[subject][sensor]["acc_max"][number]
            correlation, _ = pearsonr(x, y)
            corr.append(correlation)  # Append the correlation value to the list
            
            # print(f'The correlation for {subject} for the {sensor} in column {number +1} are {correlation}')
    # Store the correlation list in the dictionary
    correlations[subject] = corr

################ Plot that shows all of the correlation features between Min and max #########################
plt.figure()
plt.title('Correlation Values for Each Subject')
plt.ylabel('Pearson Correlation Coefficient')
for subject in subjects:    
    plt.plot(correlations[subject], label =f'person {subject}')
plt.legend()
plt.show()
# # Print the correlation lists for each subject
# for subject, corr in correlations.items():
#     print(f'The correlations for {subject} are: {corr}')