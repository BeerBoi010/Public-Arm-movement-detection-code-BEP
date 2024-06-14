import numpy as np

acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

for subject in subjects:
    for imu_location in acc[subject]:
        data_imu = [] 
        # Stacking of both acc and rot data to put through features
        acc_data_imu_split = np.array_split(acc[subject][imu_location], 381)
        rot_data_imu_split = np.array_split(rot[subject][imu_location], 381)

        # Ensure equal splits
        assert len(acc_data_imu_split) == len(rot_data_imu_split), "Mismatch in number of splits"
        
        for acc_split, rot_split in zip(acc_data_imu_split, rot_data_imu_split):
            full_data = np.hstack((acc_split, rot_split))

            for i in range(full_data.shape[1]):  # Calculate the features for each channel (column)
                step = full_data[:, i]
                Mean = np.mean(step, axis=0)
                STD = np.std(step, axis=0)
                RMS = np.sqrt(np.mean(step**2, axis=0))  # RMS value of each column
                MIN = np.min(step, axis=0)
                MAX = np.max(step, axis=0)
                window_features = np.array([Mean, STD, RMS, MIN, MAX])
                data_imu.append(window_features)
        
        # Assuming you want to store these features in a structured way
        acc[subject][imu_location] = data_imu
