import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


imu_locations = ['hand_IMU', 'lowerarm_IMU', 'upperarm_IMU', 'shoulder_IMU', 'sternum_IMU']


subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']


acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()


def compute_slope(data, sampling_window, min_periods):
    slope = data.rolling(window=sampling_window, min_periods=min_periods).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (sampling_window - 1))
    return slope


def Slope_overall(sampling_window, min_periods):
    overall_slope_data = {}

    for subject in subjects:
        acc_data_patient = acc[subject]
        rot_data_patient = rot[subject]

        for imu_location in imu_locations:
            acc_data_imu = acc_data_patient[imu_location]
            rot_data_imu = rot_data_patient[imu_location]

            dataset_acc = pd.DataFrame(acc_data_imu)
            dataset_rot = pd.DataFrame(rot_data_imu)

            slope_acc = compute_slope(dataset_acc, sampling_window, min_periods)
            slope_rot = compute_slope(dataset_rot, sampling_window, min_periods)
            

            if imu_location not in overall_slope_data:
                overall_slope_data[imu_location] = {'acc_slope': slope_acc, 'rot_slope': slope_rot}
            else:
                overall_slope_data[imu_location]['acc_slope'] = pd.concat([overall_slope_data[imu_location]['acc_slope'], slope_acc], axis=1).mean(axis=1)
                overall_slope_data[imu_location]['rot_slope'] = pd.concat([overall_slope_data[imu_location]['rot_slope'], slope_rot], axis=1).mean(axis=1)

    return overall_slope_data

# Usage example:
sampling_window = 3
min_periods = 1
overall_slope_data = Slope_overall(sampling_window, min_periods)
print(overall_slope_data)


# Function to plot the slope data
def plot_slope_data(slope_data):
    for subject, imu_data in slope_data.items():
        for imu_location, slope_values in imu_data.items():
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 1, 1)
            plt.plot(slope_values['acc_slope'], label='Acc Slope')
            plt.title(f"Slope Data for {subject} - {imu_location} (Accelerometer)")
            plt.xlabel("Sample")
            plt.ylabel("Slope")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(slope_values['rot_slope'], label='Rot Slope')
            plt.title(f"Slope Data for {subject} - {imu_location} (Gyroscope)")
            plt.xlabel("Sample")
            plt.ylabel("Slope")
            plt.legend()

            plt.tight_layout()
            plt.show()

#print(Slope_test(5,3,1))
