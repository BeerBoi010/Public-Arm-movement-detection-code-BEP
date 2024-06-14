import numpy as np
import matplotlib.pyplot as plt

annotation2 = np.load("Data_tests/Annotated times/time_ranges_subject_2.npy", allow_pickle=True)
annotation3 = np.load("Data_tests/Annotated times/time_ranges_subject_3.npy", allow_pickle=True)
annotation4 = np.load("Data_tests/Annotated times/time_ranges_subject_4.npy", allow_pickle=True)
annotation5 = np.load("Data_tests/Annotated times/time_ranges_subject_5.npy", allow_pickle=True)
annotation7 = np.load("Data_tests/Annotated times/time_ranges_subject_7.npy", allow_pickle=True)
annotation6 = np.array([
   
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
