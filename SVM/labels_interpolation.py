import numpy as np
import matplotlib.pyplot as plt

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)
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

annotation_matrices = [annotation2, annotation3, annotation4, annotation5, annotation6, annotation7]

# Define the sampling frequency
sampling_frequency = 50  #Hz
# Function to perform interpolation for a single row

def interpolate_row(row, cumulative_count):
    # Convert start and end time to floats
    start_time = float(row[0])
    end_time = float(row[1])
    # Original label
    label = row[2]
    # Calculate the number of samples
    num_samples = round((end_time - start_time) * sampling_frequency)
    # Create expanded rows with data points and label starting from index 0
    expanded_rows = [[cumulative_count + i, label] for i in range(num_samples)]
    # Update cumulative count
    cumulative_count += num_samples
    return expanded_rows, cumulative_count

# Initialize list to store expanded rows for all participants
expanded_matrices = []

# Iterate over each participant's annotation matrix
for annotation_matrix in annotation_matrices:
    # Initialize variables to keep track of the cumulative count
    cumulative_count = 0
    # Initialize list to store expanded rows for the current participant
    expanded_matrix = []
    # Iterate over each row in the annotation matrix and perform interpolation
    for row in annotation_matrix:
        expanded_rows, cumulative_count = interpolate_row(row, cumulative_count)
        expanded_matrix.extend(expanded_rows)
    # Append the expanded matrix for the current participant to the list
    expanded_matrices.append(expanded_matrix)

exp_annotations2 = expanded_matrices[0]
exp_annotations3 = expanded_matrices[1]
exp_annotations4 = expanded_matrices[2]
exp_annotations5 = expanded_matrices[3]
exp_annotations6 = expanded_matrices[4]
exp_annotations7 = expanded_matrices[5]
#print(expanded_matrices)
print(exp_annotations2)
