import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

#### description: first try implementing Arkady's ideas for cross-over zones and implementing them. 
###Rest is still similar to V1.



#important variables:
sample_size = 3

# Define IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']

# Iterate over each subject and IMU location
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Load the .npy files
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation6 = np.load("Data_tests/Annotated times/time_ranges_subject_6.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

# Define the label mapping dictionary
label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}



# Function to handle crossover zones using majority voting

# Function to handle crossover zones using majority voting
def handle_crossover_zone(labels):
    window_size = 3
    new_labels = []
    for i in range(len(labels) - window_size + 1):
        window = labels[i:i + window_size]
        counts = np.bincount(window.astype(int), minlength=4)  # Ensure counts have length equal to number of classes
        majority_label = np.argmax(counts)  # Majority voting
        new_labels.append(majority_label)
    
    # Handle labels at the end of the array
    for i in range(len(labels) - window_size + 1, len(labels)):
        window = labels[i:]
        counts = np.bincount(window.astype(int), minlength=4)  # Ensure counts have length equal to number of classes
        majority_label = np.argmax(counts)  # Majority voting
        new_labels.append(majority_label)
    
    return new_labels


# Process annotations and map labels
annotation_mapping = {
    'drinking_HealthySubject2_Test': annotation2,
    'drinking_HealthySubject3_Test': annotation3,
    'drinking_HealthySubject4_Test': annotation4,
    'drinking_HealthySubject5_Test': annotation5,
    'drinking_HealthySubject6_Test': annotation6,
    'drinking_HealthySubject7_Test': annotation7
}

mapped_labels = {}
for subject, annotation in annotation_mapping.items():
    mapped_labels[subject] = np.array([[item[0], item[1], label_mapping[item[2]]] for item in annotation])
    mapped_labels[subject][:, 2] = handle_crossover_zone(mapped_labels[subject][:, 2])

# Process data for training
X_data_patients_train = []
labels_patients_train = []

for subject in subjects[:4]:
    acc_data_patient = acc[subject]
    rot_data_patient = rot[subject]
    labels_patient = mapped_labels[subject][:, 2].astype(int)

    print(labels_patient)
    print(labels_patient.shape)

    # for row in annotation[subject]:
    #     label = int(row[2])
    #     start_time = float(row[0])
    #     end_time = float(row[1])
    #     duration = end_time - start_time
    #     num_measurements = round(duration * Hz)
    #     #print("variables",start_time,end_time,label,duration,num_measurements)
    #     labels_patient.extend([label] * num_measurements)
    
    # if subject == 'drinking_HealthySubject6_Test':
    #     labels_patient = labels_patient[:-5]  # Delete the last 5 labels
    
    combined_data_patient = []
    for imu_location in imu_locations:
        acc_data_imu = acc_data_patient[imu_location]
        rot_data_imu = rot_data_patient[imu_location]
        
        combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
        combined_data_patient.extend(combined_data_imu.T)
    
    X_data_patients_train.append(np.vstack(combined_data_patient).T)
    labels_patients_train.append(labels_patient)

# Combine data and labels from all patients
combined_X_data = np.concatenate(X_data_patients_train)
combined_labels = np.concatenate(labels_patients_train)

#print(combined_labels)
#print(combined_X_data.shape, combined_labels.shape)
