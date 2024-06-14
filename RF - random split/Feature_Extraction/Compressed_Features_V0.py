import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Feature_Extraction import RMS_V2, Mean_V2, Slope_V2, Max_V2, Min_V2, Standard_Deviation
import labels_interpolation

train_amount = 6
sampling_window = 3
min_periods = 1
test_amount = train_amount

sampling_window_RMS = 3
sampling_window_min_max = 3
sampling_window_mean = 3
sampling_window_STD = 3
sampling_window_slope = 3
test_person = 7
print(f'drinking_HealthySubject{test_person}_Test')
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

subjects.remove(f'drinking_HealthySubject{test_person}_Test')
subjects_train = subjects
subjects_test = [f'drinking_HealthySubject{test_person}_Test']

#%% Extracting features: 
window_length_sec = 0.5 # 0.5 second
overlap = 0.5
windows_AllSubject = []
Labels_AllSubject = []
window_counts_AllSubject = [] 
# To fix the length of windows for each subject as the same, 
# Define the number of features
num_features = 15  # 7 statistical features per channel * 3 channels + 3 correlation coefficients

List_1= np.split(acc['drinking_HealthySubject2_Test'])


# # Initialize Feature_Matrix
# Feature_Matrix = np.zeros((0, num_features)) 

# for subject_windows_data in windows_AllSubject:
#     subject_feature_matrix = np.zeros((0, num_features)) #subject_feature_matrix has same number of rows as the number of windows and same number of colmuns as the number of Feature*Channels.
    
#     for seg in subject_windows_data:
#         # Calculate statistical features for each channel(Column)
#         Mean = np.mean(seg, axis=0)
#         STD = np.std(seg, axis=0)
#         RMS = np.sqrt(np.mean(seg**2, axis=0))  #RMS value of each column.
#         MIN = np.min(seg, axis=0)
#         MAX = np.max(seg, axis=0)
        
#         window_features = np.concatenate((Mean, STD, RMS, MIN, MAX))
        
#         # Append the features for the current window to subject_feature_matrix
#         subject_feature_matrix = np.vstack((subject_feature_matrix, window_features))
    
#     # Append the features for the current subject to Feature_Matrix
#     Feature_Matrix = np.vstack((Feature_Matrix, subject_feature_matrix))

