
# CNN CODE made with ChatGPT. Does not work

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter

import tensorflow_addons as tfa
import labels_interpolation

# Load data
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()
all_labels = labels_interpolation.expanded_matrices

subjects = ['drinking_HealthySubject2_Test', 'drinking_HealthySubject3_Test', 'drinking_HealthySubject4_Test',   
            'drinking_HealthySubject5_Test', 'drinking_HealthySubject6_Test', 'drinking_HealthySubject7_Test']

# Prepare subjects and labels
# Parameters
bin_size = 50
overlap = 0

def majority_vote(labels):
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

def process_labels(labels, bin_size, overlap):
    step = bin_size - overlap
    processed_labels = []
    for start in range(0, len(labels) - bin_size + 1, step):
        window = labels[start:start + bin_size]
        majority_label = majority_vote([label[1] for label in window])
        processed_labels.append(majority_label)
    return processed_labels

# Process labels
labels_train = []
for item in all_labels:
    labels_train.extend(process_labels(item, bin_size, overlap))

label_mapping = {'N': 0, 'A': 1, 'B': 2, 'C': 3}
y_train = [label_mapping[label] for label in labels_train]

print("labels",y_train,len(y_train))

# One-hot encode labels
y_train_oh = to_categorical(y_train)

# Reshape the labels to match the desired shape
y_train_oh = y_train_oh.reshape((-1, bin_size, y_train_oh.shape[-1]))

# Prepare the training and test data
def prepare_raw_data(subjects, acc, rot, bin_size, overlap):
    data = []
    for subject in subjects:
        subject_data = []
        for imu_location in acc[subject]:
            acc_data = acc[subject][imu_location]
            min_val = np.min(acc_data, axis=None, keepdims=True)
            max_val = np.max(acc_data, axis=None, keepdims=True)
            normalized_acc = (acc_data - min_val) / (max_val - min_val) * 2 - 1

            rot_data = rot[subject][imu_location]
            min_val = np.min(rot_data, axis=None, keepdims=True)
            max_val = np.max(rot_data, axis=None, keepdims=True)
            normalized_rot = (rot_data - min_val) / (max_val - min_val) * 2 - 1

            imu_data = np.hstack((normalized_acc, normalized_rot))
            subject_data.append(imu_data)
        subject_data = np.hstack(subject_data)
        data.append(subject_data)
    return np.array(data)

X_train_raw = prepare_raw_data(subjects, acc, rot, bin_size, overlap)

# Train-test split
X_train_raw, X_test_raw, y_train_oh, y_test_oh = train_test_split(X_train_raw, y_train_oh, test_size=0.2, random_state=42)

input_shape = (X_train_raw.shape[1], X_train_raw.shape[2])
output_shape = (y_train_oh.shape[1], y_train_oh.shape[2])

# Define CNN model
def create_cnn_model(input_shape, output_shape, l2_lambda=0.1):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_lambda)),
        MaxPooling1D(2),
        Dropout(0.6),
        Conv1D(64, 3, activation='relu', kernel_regularizer=l2(l2_lambda)),
        MaxPooling1D(2),
        Dropout(0.6),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(0.6),
        Dense(output_shape[0] * output_shape[1], activation='softmax'),
        Reshape(output_shape)
    ])
    return model

model = create_cnn_model(input_shape, output_shape, l2_lambda=0.01)
optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_raw, y_train_oh, epochs=300, batch_size=1, validation_data=(X_test_raw, y_test_oh), callbacks=[early_stopping])

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_raw, y_test_oh)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Predict on test data
y_test_pred = model.predict(X_test_raw)
y_test_pred_classes = np.argmax(y_test_pred, axis=2)
y_test_true_classes = np.argmax(y_test_oh, axis=2)

print("Classification Report of test data:")
print(classification_report(y_test_true_classes[0], y_test_pred_classes[0], zero_division=1))

conf_matrix = confusion_matrix(y_test_true_classes[0], y_test_pred_classes[0])
label_mapping_inv = {v: k for k, v in label_mapping.items()}

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[label_mapping_inv[key] for key in range(output_shape[1])], yticklabels=[label_mapping_inv[key] for key in range(output_shape[1])])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
