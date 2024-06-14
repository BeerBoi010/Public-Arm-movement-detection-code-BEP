import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def load_annotations():
    annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
    annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
    annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
    annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
    annotation6 = np.array([
        ['0', '1.4', 'N'],
        ['1.4', '2.6', 'A'],
        ['2.6', '4.8', 'B'],
        ['4.8', '6', 'C'],
        ['6', '7.5', 'B'],
        ['7.5', '8.8', 'A'],
        ['8.8', '11.3', 'N'],
        ['11.3', '12.6', 'A'],
        ['12.6', '13.6', 'B'],
        ['13.6', '14.3', 'C'],
        ['14.3', '15.7', 'B'],
        ['15.7', '17.6', 'A'],
        ['17.6', '18.4', 'N'],
        ['18.4', '19.2', 'A'],
        ['19.2', '20.3', 'B'],
        ['20.3', '21.3', 'C'],
        ['21.3', '22.5', 'B'],
        ['22.5', '23.5', 'A'],
        ['23.5', '24.5', 'N'],
        ['24.5', '25.4', 'A'],
        ['25.4', '26.7', 'B'],
        ['26.7', '27.7', 'C'],
        ['27.7', '29', 'B'],
        ['29', '30', 'A'],
        ['30', '31.4', 'N'],
        ['31.4', '32.1', 'A'],
        ['32.1', '33.3', 'B'],
        ['33.3', '34.4', 'C'],
        ['34.4', '35.8', 'B'],
        ['35.8', '37', 'A'],
        ['37', '38.1', 'N']
    ])
    annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)

    annotations = {
        'drinking_HealthySubject2_Test': annotation2,
        'drinking_HealthySubject3_Test': annotation3,
        'drinking_HealthySubject4_Test': annotation4,
        'drinking_HealthySubject5_Test': annotation5,
        'drinking_HealthySubject6_Test': annotation6,
        'drinking_HealthySubject7_Test': annotation7
    }

    label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}
    
    mapped_annotations = {}
    for subject, annotation in annotations.items():
        mapped_annotations[subject] = np.array([[item[0], item[1], label_mapping[item[2]]] for item in annotation])

    return mapped_annotations

def preprocess_data(subjects, imu_locations, annotations, acc, rot):
    X_data = []
    labels = []

    for subject in subjects:
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]
        labels_patient = [] 

        for row in annotations[subject]:
            label = int(row[2])
            start_time = float(row[0])
            end_time = float(row[1])
            duration = end_time - start_time
            num_measurements = round(duration * Hz)
            labels_patient.extend([label] * num_measurements)

        combined_data_patient = []
        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]
            combined_data_imu = np.hstack((acc_data_imu, rot_data_imu))
            combined_data_patient.extend(combined_data_imu.T)
        
        X_data.append(np.vstack(combined_data_patient).T)
        labels.append(labels_patient)

    combined_X_data = np.concatenate(X_data)
    combined_labels = np.concatenate(labels)

    return combined_X_data, combined_labels

def train_and_evaluate_model(train_subjects, test_subjects, imu_locations):
    # Load the data
    acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
    rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
    annotations = load_annotations()

    # Preprocess training data
    X_train, y_train = preprocess_data(train_subjects, imu_locations, annotations, acc, rot)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Preprocess test data
    X_test, y_test = preprocess_data(test_subjects, imu_locations, annotations, acc, rot)
    X_test = scaler.transform(X_test)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    # Evaluate the model
    print("Classification Report of train data:")
    print(classification_report(y_train, y_train_pred))

    print("Classification Report of test data:")
    print(classification_report(y_test, y_test_pred))

    # Plot results for each test subject
    test_lengths = [len(annotations[subject]) for subject in test_subjects]
    split_y_pred = np.split(y_test_pred, np.cumsum(test_lengths)[:-1])
    split_y_test = np.split(y_test, np.cumsum(test_lengths)[:-1])

    for i, subject in enumerate(test_subjects):
        y_pred_patient = split_y_pred[i]
        y_test_patient = split_y_test[i]
        element_numbers = list(range(len(y_pred_patient)))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(element_numbers, y_pred_patient, label='Predictions', color='blue')
        plt.xlabel('Element Numbers')
        plt.ylabel('Predicted Labels')
        plt.title(f'Predicted Labels - {subject}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(element_numbers, y_test_patient, label='True Labels', color='green')
        plt.xlabel('Element Numbers')
        plt.ylabel('True Labels')
        plt.title(f'True Labels - {subject}')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance")
    plt.show()

# Define subjects and IMU locations
imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']
subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test', 'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Define training and test subjects
train_subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',
                   'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test']
test_subjects = ['drinking_HealthySubject7_Test']

# Calculate sampling frequency
x_acceleration = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()['drinking_HealthySubject2_Test']['hand_IMU']
Hz = len(x_acceleration) / 38.1

# Train and evaluate the model
train_and_evaluate_model(train_subjects, test_subjects, imu_locations)
