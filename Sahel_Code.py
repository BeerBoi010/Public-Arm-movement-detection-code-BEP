# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:59:52 2023

@author: sahel
"""

        

##reamining _tasks: 
## Better features? (find what type of features are performing the best)
# Try to find a method for estimating the movement duration. 
# calculate the same feature matrix with the acc signal as well? (Done)
# consider adding the joint angels and quartenion data, 
# annonate the other subject data, leave one out subject: test and train: (Done)
# Build generalized models, (Done) 
# ROC curve, 
# try CNN model 
# Different Models? trying models that are independent to the length of signal.
# Different sensors accuracy?  
# build on healthy data, test on stroke data?
# Do I care about the sequence of Labels or the individual label for each window?(TimeSeriesSplit---> does not work 

#%% 

## Main code to read data, implement general preprocessing steps, 
## extract relevant features and to build SVM, RF and CNN and LSTM models, 
## for more information please refer to " The pragmatic classification of upper extrimity motions in neurological patients, a primer"

#%% Import main modules: 
    
import matplotlib.pyplot as plt
import numpy as np 
import os 
from scipy.stats import skew, kurtosis
from scipy.stats import pearsonr

       
Arm_side = 'R'         # 'R' for right body side, 'L' for left body side.
fs= 50              # Sampling frequency

#%% Import PreProcessed data: 
# wdir= r"C:\Users\Mieke\Documents\dir_machinelearning"
# os.chdir(wdir)
data = np.load("Data_tests/data_Preprocessed.npy", allow_pickle='True').item()
Gyro_signal= np.load("Data_tests/ACC_signal.npy",allow_pickle='TRUE').item()
Acc_signal = np.load("Data_tests/ACC_signal.npy",allow_pickle='TRUE').item()

#%% Ground truth labels: 
# Define labels for the all Subjects Downsample data: (Subjects 2,3,4,5,6 for training the model)  
# Five states are defined (A-Reach, B-Lift, C-Rotation of the wrist, D-Grasping, N-resting periods) 
time_ranges_subject_2 = [(0,1.5, 'N'), (1.5,2.6, 'A'), (2.6, 3.4, 'B'), (3.4, 4.7, 'C'), (4.7, 6.1, 'B'),(6.1, 7.5, 'A'), (7.5, 10.4, 'N'), (10.4, 11.3, 'A'), 
               (11.3, 12.3,'B'), (12.3,13.3, 'C'),(13.3,14.7, 'B'),(14.7,15.9, 'A'),(15.9,17.8,  'N'),(17.8,18.7, 'A'),(18.7,19.6, 'B'),(19.6,20.6, 'C'),
               (20.6,21.8, 'B'),(21.8,23.2, 'A'),(23.2,25, 'N'),(25,25.8, 'A'),(25.8,26.8, 'B'),(26.8,27.7, 'C'),(27.7,29, 'B'), (29,30.5, 'A'),
               (30.5,31.7, 'N'),(31.7,32.6, 'A'),(32.6,33.5, 'B'),(33.5, 34.4, 'C'),(34.4,35.7, 'B'),(35.7,37, 'A'),(37,38.1, 'N')]

time_ranges_subject_3=[(0,1.5, 'N'), (1.5, 2.6, 'A'), (2.6, 3.8, 'B'), (3.8, 4.8, 'C'), (4.8, 6.3, 'B'),(6.3, 7.6, 'A'), (7.6, 10.8, 'N'), (10.8, 11.8, 'A'), 
               (11.8, 13,'B'), (13,13.6, 'C'),(13.6,14.9, 'B'),(14.9,16, 'A'),(16,17.8, 'N'),(17.8,18.6, 'A'),(18.6,19.6, 'B'),(19.6,20.3, 'C'),
               (20.3,21.9, 'B'),(21.9,23, 'A'),(23,24.7, 'N'),(24.7,25.4, 'A'),(25.4,26.7, 'B'),(26.7,27.5, 'C'),(27.5,28.5, 'B'), (28.5,29.7, 'A'),
               (29.7, 31.3, 'N'),(31.3, 32.2, 'A'),(32.2,33.3, 'B'),(33.3,34.2, 'C'),(34.2,35.7, 'B'),(35.7,36.7, 'A'),(36.7,38.1, 'N')] 
               
time_ranges_subject_4=[(0,1.2, 'N'), (1.2, 2.2, 'A'), (2.2, 3.1, 'B'), (3.1, 3.8, 'C'), (3.8, 5.4, 'B'),(5.4, 7, 'A'), (7, 9.8, 'N'), (9.8, 10.7, 'A'), 
               (10.7, 11.7,'B'), (11.7,12.3, 'C'),(12.3,13.6, 'B'),(13.6,14.7, 'A'),(14.7,17.8, 'N'),(17.8,18.8, 'A'),(18.8,19.8, 'B'),(19.8,20.4, 'C'),
               (20.4,21.8, 'B'),(21.8,22.9, 'A'),(22.9,25.9, 'N'),(25.9,26.6, 'A'),(26.6,27.7, 'B'),(27.7,28.5, 'C'),(28.5,29.7, 'B'), (29.7,31.2, 'A'),
               (31.2,33, 'N'),(33,33.7, 'A'),(33.7,34.6, 'B'),(34.6,35.2, 'C'),(35.2,36.2, 'B'),(36.2,37.2, 'A'),(37.2,38.1, 'N')]

time_ranges_subject_5=[(0,1.5, 'N'), (1.5, 2.3, 'A'), (2.3, 3.5, 'B'), (3.5, 4, 'C'), (4, 5.8, 'B'),(5.8,6.7 , 'A'), (6.7, 9.1, 'N'), (9.1, 9.9, 'A'), 
               (9.9, 10.9,'B'), (10.9,11.5, 'C'),(11.5,12.5, 'B'),(12.5,13.9, 'A'),(13.9,15.9, 'N'),(15.9,16.5, 'A'),(16.5,17.5, 'B'),(17.5,18.4, 'C'),
               (18.4,19.8, 'B'),(19.8,21.2, 'A'),(21.2,23.5, 'N'),(23.5,24.3, 'A'),(24.3,25.5, 'B'),(25.5,26.1, 'C'),(26.1,27.5, 'B'), (27.5,29, 'A'),
               (29,31.8, 'N'),(31.8,32.6, 'A'),(32.6,33.4, 'B'),(33.4,34.3, 'C'),(34.3,35.7, 'B'),(35.7,37.1, 'A'),(37.1,38.1, 'N')] 

time_ranges_subject_6=[(0,1.4, 'N'), (1.4, 2.6, 'A'), (2.6, 4.8, 'B'), (4.8, 6, 'C'), (6, 7.5, 'B'),(7.5, 8.8, 'A'), (8.8, 11.3, 'N'), (11.3, 12.6, 'A'), 
               (12.6, 13.6,'B'), (13.6,14.3, 'C'),(14.3,15.7, 'B'),(15.7,17.6, 'A'),(17.6,18.4, 'N'),(18.4,19.2, 'A'),(19.2,20.3, 'B'),(20.3,21.3, 'C'),
               (21.3,22.5, 'B'),(22.5,23.5, 'A'),(23.5,24.5, 'N'),(24.5,25.4, 'A'),(25.4,26.7, 'B'),(26.7, 27.7, 'C'),(27.7,29, 'B'), (29,30, 'A'),
               (30,31.4, 'N'),(31.4, 32.1, 'A'),(32.1,33.3, 'B'),(33.3,34.4, 'C'),(34.3,35.8, 'B'),(35.8, 37, 'A'),(37,38.1, 'N')]
            

time_ranges_subject_7=[(0,1.8, 'N'), (1.8, 2.6, 'A'), (2.6, 4, 'B'), (4, 4.8, 'C'), (4.8, 6.5, 'B'),(6.5, 7.7, 'A'), (7.7, 9.5, 'N'), (9.5, 10.3, 'A'), 
               (10.3, 11.4,'B'), (11.4,12.1, 'C'),(12.1,13.8, 'B'),(13.8,15.2, 'A'),(15.2,16.9, 'N'),(16.9,17.5, 'A'),(17.5,18.8, 'B'),(18.8,19.5, 'C'),
               (19.5,21.1, 'B'),(21.1,22.3, 'A'),(22.3,23.7, 'N'),(23.7,24.5, 'A'),(24.5,25.8, 'B'),(25.8,26.6, 'C'),(26.6,28, 'B'), (28,29.3, 'A'),
               (29.3,31, 'N'),(31,31.8, 'A'),(31.8,32.9, 'B'),(32.9,33.5, 'C'),(33.5,35, 'B'),(35,36.5, 'A'),(36.5,38.1, 'N')] 

#stack segmentation for all subjects together: 
subjects_time_ranges = [time_ranges_subject_2 , time_ranges_subject_3, time_ranges_subject_4 ,time_ranges_subject_5, 
                time_ranges_subject_6, time_ranges_subject_7 ]

#Stack ACC-Hand IMU data for all subjects together: 
subjects_data = [Acc_signal['drinking_HealthySubject2_Test']['hand_IMU'], Acc_signal['drinking_HealthySubject3_Test']['hand_IMU'], 
                Acc_signal['drinking_HealthySubject4_Test']['hand_IMU'], Acc_signal['drinking_HealthySubject5_Test']['hand_IMU'], 
                Acc_signal['drinking_HealthySubject6_Test']['hand_IMU'], Acc_signal['drinking_HealthySubject7_Test']['hand_IMU']] 

np.save('time_ranges_subject_2.npy', time_ranges_subject_2)
np.save('time_ranges_subject_3.npy', time_ranges_subject_3)
np.save('time_ranges_subject_4.npy', time_ranges_subject_4)
np.save('time_ranges_subject_5.npy', time_ranges_subject_5)
np.save('time_ranges_subject_6.npy', time_ranges_subject_6)
np.save('time_ranges_subject_7.npy', time_ranges_subject_7)

#%% This piece of code, segments each subject data according to their labels, including the same length windows: 

window_length_sec = 50 # 0.5 second
overlap = 0.5 #0.5
window_length_samples = int(window_length_sec * fs)
windows_AllSubject = []
Labels_AllSubject = []
window_counts_AllSubject = [] 


for subject_data, time_range in zip(subjects_data, subjects_time_ranges):
    labeled_windows = []
    windows_data=[]
    labels = []
    window_counts_part = []

#Segment each part according to the time ranges with fixed window length and overlap
    for start, end, label in time_range:
        start_idx = int(start * fs)
        end_idx = int(end * fs)
        part_duration = end - start
        num_windows = int((part_duration - window_length_sec) / (window_length_sec * (1 - overlap))) + 1 # Calculate the number of windows with 50% overlap
        window_counts_part.append(num_windows)
        for i in range(num_windows):
            window_start = start_idx + i * int(window_length_samples * overlap)
            #window_end = min(window_start + window_length_samples, end_idx)
            window_end = window_start + window_length_samples
            window = subject_data[window_start:window_end]
            labeled_windows.append((label, window))

    # Append only windows that have the same length
    for label, window in labeled_windows:
        if len(window) >= 25:  # eliminate windows that have shorter length, 
            windows_data.append(window)
            labels.append(label)
        else: 
             num_windows=-1 
    
    windows_AllSubject.append(windows_data)
    Labels_AllSubject.append(labels)
    window_counts_AllSubject.append(window_counts_part)
    
# To fix the length of windows for each subject as the same, 
min_length = min(len(subject_windows_data) for subject_windows_data in windows_AllSubject)
windows_AllSubject = [subject_windows_data[:min_length] for subject_windows_data in windows_AllSubject]
Labels_AllSubject = [subject_labels[:min_length] for subject_labels in Labels_AllSubject]


#%% Extracting features: 
    
# Define the number of features
num_features = 24  # 7 statistical features per channel * 3 channels + 3 correlation coefficients

# Initialize Feature_Matrix
Feature_Matrix = np.zeros((0, num_features)) 

for subject_windows_data in windows_AllSubject:
    subject_feature_matrix = np.zeros((0, num_features)) #subject_feature_matrix has same number of rows as the number of windows and same number of colmuns as the number of Feature*Channels.
    
    for seg in subject_windows_data:
        # Calculate statistical features for each channel(Column)
        Mean = np.mean(seg, axis=0)
        STD = np.std(seg, axis=0)
        RMS = np.sqrt(np.mean(seg**2, axis=0))  #RMS value of each column.
        MIN = np.min(seg, axis=0)
        MAX = np.max(seg, axis=0)
        KURTOSIS = kurtosis(seg, axis=0)  #a measure of ditribution of the data, >0 many outliers and high peaks, <0 low peaks and less outliers 
        SKEWNESS = skew(seg, axis=0) # a measure of symmetry of the data, >0 tail on the right, <0 tail on the left 
        
        # Calculate correlation coefficients between channels
        # Calculating the corr between channels can help to remove reduandant features, (if two channels are highly corrolated, then the feature space can be minimized)
        CorrCoefXY, _ = pearsonr(seg[:, 0], seg[:, 1])
        CorrCoefYZ, _ = pearsonr(seg[:, 1], seg[:, 2])
        CorrCoefXZ, _ = pearsonr(seg[:, 0], seg[:, 2])
        
        window_features = np.concatenate((Mean, STD, RMS, MIN, MAX, KURTOSIS, SKEWNESS, [CorrCoefXY, CorrCoefYZ, CorrCoefXZ]))
        
        # Append the features for the current window to subject_feature_matrix
        subject_feature_matrix = np.vstack((subject_feature_matrix, window_features))
    
    # Append the features for the current subject to Feature_Matrix
    Feature_Matrix = np.vstack((Feature_Matrix, subject_feature_matrix))

#%% Improving Accuracy: 

# 1- feature reduction: (makes it worse for now)
from sklearn.decomposition import PCA
n_components = 10 # Number of components to keep
pca = PCA(n_components=n_components)
pca.fit(Feature_Matrix)
Feature_Matrix_pca = pca.transform(Feature_Matrix)


# 1-prime: Feature reduction using LDA: (better accuracy results with LDA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
n_components = 3  # Number of components to keep depends on the number of classes-1
lda = LinearDiscriminantAnalysis(n_components=n_components)   
labels = np.concatenate(Labels_AllSubject)
lda.fit(Feature_Matrix, labels)
feature_importance = lda.scalings_   # in feature importance each row corresponds to the a feature and each colmun to a linear discriminants for that feature. 
absolute_importance = np.abs(feature_importance)
strongest_features_indices = np.argsort(absolute_importance, axis=0)[::-1]

#print the strongest features that have the most influence in building each discriminants: 
for component in range(n_components):
    print(f"Discriminant {component + 1}:")
    for i, feature_index in enumerate(strongest_features_indices[:, component]):
        feature_importance_value = feature_importance[feature_index, component]
        print(f"  Feature {feature_index}: Importance = {feature_importance_value}")

# transform Feature matrix in the lda discriminant space: 
Feature_Matrix_lda = lda.transform(Feature_Matrix)
 
#%% 
    
# 2- Feature Scaling: this part will scale each feature (column) across the whole samples: 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(Feature_Matrix_pca)
Feature_Matrix_scaled = scaler.transform(Feature_Matrix_pca) 


#%% Ruuning the Models (RF, SVM, LDA) 
## Split subjects for test and train, (OneSubjectleaveOut) 

from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np


np.random.seed(42)
labels = np.concatenate(Labels_AllSubject)
class_labels =['A', 'B', 'C', 'N']

# Define LeaveOneGroupOut cross-validation for each subject: 
logo_outer = LeaveOneGroupOut()
Allaccuracies = []
Test_allSubjectLabels=[]
subjectID=2
# Iterate over each subject for leave-one-subject-out:
for train_index, test_index in logo_outer.split(Feature_Matrix_scaled, labels, groups=np.concatenate([[i]*len(l) for i, l in enumerate(Labels_AllSubject)])):
    print(f"Subject {subjectID}:")
    X_train, X_test = Feature_Matrix_scaled[train_index], Feature_Matrix_scaled[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
     
    rf_classifier = RandomForestClassifier()  # Initialize RandomForestClassifier each time for each subject
    rf_classifier.fit(X_train, y_train)
    y_pred_test = rf_classifier.predict(X_test)
    
    # svm_classifier = SVC(kernel='rbf', C=10, random_state=42)  # Initialize SVM model
    # svm_classifier.fit(X_train, y_train)
    # y_pred_test = svm_classifier.predict(X_test)
    Test_allSubjectLabels.append(y_pred_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    Allaccuracies.append(accuracy_test)
    print(f"Accuracy for Subject {subjectID}: {accuracy_test:.4f}")
    print("Class Labels:", class_labels)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)

    # Normalize the confusion matrix to get percentages
    conf_matrix_percent = conf_matrix / row_sums * 100
    
    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_percent, annot=True, cmap='Blues', fmt='g', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    # plt.title(f'Confusion Matrix for Subject {subjectID}')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    
    subjectID += 1

# Calculate average accuracy across all subjects
average_accuracy = np.mean(Allaccuracies)

#%% A piece of code to calculate each movement duration according to the labels that were predicted: 
## What are the majority labels that were predicted? 
## How many of windows inside of each movement segment were predicted accurately? what is the accuracy? 

        
import statistics

Majority_Vote_subject = []

for subject_ind, part in enumerate(window_counts_AllSubject):
    Majority_Vote_windows = []  # List to store majority votes for windows of the current subject
    start_index=0
    for length in part: 
        end_index = start_index + length-1 
        if start_index== end_index:
              sliced_labels= Test_allSubjectLabels[subject_ind][start_index]
        else: 
              sliced_labels = Test_allSubjectLabels[subject_ind][start_index:end_index]
            
        
        Majority_Vote = statistics.mode(sliced_labels)
        Majority_Vote_windows.append(Majority_Vote)
        start_index = end_index
    
    Majority_Vote_subject.append(Majority_Vote_windows)
    print (  f"The majority vote for subject {subject_ind} per window is {Majority_Vote_subject[subject_ind]}")

        

#%% CNN running in allsubjects data, inputs are same length time series-windows: 
## https://datascience.stackexchange.com/questions/64022/why-must-a-cnn-have-a-fixed-input-size
## each CNN has three main blocks: Convlution layer, Pooling layer and a fully connected layer.  


# from sklearn.model_selection import LeaveOneGroupOut
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Dropout
# from sklearn.metrics import accuracy_score



# # Convert the lists to numpy arrays for easier manipulation
# windows_AllSubject_array = np.array(windows_AllSubject)
# Labels_AllSubject_array = np.array(Labels_AllSubject)

# logo_outer = LeaveOneGroupOut()
# accuracies_CNN=[]

# # Iterate over each subject for leave-one-subject-out
# for train_index, test_index in logo_outer.split(windows_AllSubject_array, groups=np.arange(len(windows_AllSubject_array))):
    
#     X_train, X_test = windows_AllSubject_array[train_index], windows_AllSubject_array[test_index]
#     y_train, y_test = Labels_AllSubject_array[train_index], Labels_AllSubject_array[test_index]
    
#     # Print the group identifiers for training and test subjects
#     train_groups = np.arange(len(windows_AllSubject_array))[train_index]
#     test_group = np.arange(len(windows_AllSubject_array))[test_index]
#     print("Training Groups:", train_groups)
#     print("Test Group:", test_group)
    
#     # Flatten the windows dats % The faltten is because the input to the fully connceted Convolution must be only one dimension. 
#     X_train_flat = X_train.reshape(X_train.shape[0], -1)
#     X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
# # Define the Fully Connected CNN model architecture
#     model = Sequential()
#     model.add(Dense(128, activation='relu', input_shape=(X_train_flat.shape[1],)))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4, activation='softmax'))  # Assuming 4 classes
    
#     # Compile the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     # Train the model
#     model.fit(X_train_flat, y_train, epochs=10, batch_size=32, verbose=0)
    
#     # Evaluate the model on the test data
#     _, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
#     accuracies_CNN.append(test_accuracy)

# # Calculate average accuracy across all subjects
# average_accuracy = np.mean(accuracies_CNN)
# print("Average Test Accuracy:", average_accuracy)
    
    
    
#%% Give input (each movement with corresponding labels) with different length and run a LSTM model:

# # https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes, 
# # including one extra layer in NN that can handle different length. Like LSTM and padding and masking 

# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.losses import SparseCategoricalCrossentropy


# num_folds = 10
# epochs = 50
# batch_size = 64
# np.random.seed(42)
# all_test_loss=[] 
# all_test_accuracy=[]
# all_predicted_labels=[] 
# all_true_labels=[]


# # Reshape signals to (samples, signal length, number of variables)
# windows_data_array = np.array(windows_data)

# # Encode labels to integers
# label_encoder = LabelEncoder()
# numerical_labels = label_encoder.fit_transform(Labels_AllSubject[5])

# # Initialize k-fold cross-validation
# kf = KFold(n_splits=num_folds, shuffle=False)

# # Loop over each fold
# for fold, (train_index, test_index) in enumerate(kf.split(windows_data_array)):
#     print(f"Fold {fold + 1}/{num_folds}")
    
#     # Split data into training and testing sets for this fold
#     X_train, X_test = windows_data_array[train_index], windows_data_array[test_index]
#     y_train, y_test = numerical_labels[train_index], numerical_labels[test_index]
    
#     # Define and compile the LSTM model
#     model = Sequential()
#     model.add(LSTM(units=50, input_shape=(5, 3)))
#     model.add(Dense(units=5, activation='softmax'))
#     model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
#     # Train the model
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
#     # Evaluate the model on the test set
#     test_loss, test_accuracy = model.evaluate(X_test, y_test)
#     all_test_loss.append(test_loss)
#     all_test_accuracy.append(test_accuracy)
#     print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
    
#     #Predict model for testing data: 
#     predictions = model.predict(X_test)
#     predicted_labels = np.argmax(predictions, axis=1)
    
#     # Store predicted labels and true labels for this fold
#     all_predicted_labels.append(predicted_labels)
#     all_true_labels.append(y_test)
    

# #%% try to handle different length of data with the LSTM + padding and Masking technique: 
# labeled_data=[]

# for start, end, label in time_ranges_subject_2: 
#     data = time_series_signal[int(start * fs) : int(end * fs), :]
#     labeled_data.append((label, data))
        
# data_all = [data for _, data in labeled_data]
# labels_all = [label for label, _ in labeled_data]
# #data_all_array = np.array(data_all)


# #find the max length of a part/section of signal to use for padding:  
# Max_len = max(len(i) for i in data_all)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Masking
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# ################################# the next part is still wrong: 

# #Padd data: 
# padded_data = pad_sequences(
#     data_all, maxlen=Max_len, dtype="int32", padding="post", truncating="post", value=0.0)

# # Define the model architecture
# model = Sequential()

# #Add a Masking layer to handle sequences with different lengths
# model.add(Masking(mask_value=0.0, input_shape=(None, 3)))

# # Add the LSTM layer
# model.add(LSTM(units=50))

# # Add the output Dense layer
# model.add(Dense(units=5, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# # Print model summary
# model.summary()




    


