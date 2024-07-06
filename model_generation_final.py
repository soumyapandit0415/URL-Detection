import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


# Configuring Dataset and Values

# Reading in dataset from CSV file. This dataset is an updated version of the original Kaggle dataset including 
urldata = pd.read_csv("./Url_Processed.csv")

# Clean up dataset and remove unecessary columns
urldata.drop("Unnamed: 0",axis=1,inplace=True)
urldata.drop(["url","label"],axis=1,inplace=True)

# Configure dependent variables (values used to inform prediction)
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

# Configure independent variable (value to verify prediction)
y = urldata['result']

# Using SMOTE to resample dataset. The SMOTE (Synthetic Minority Over-sampling Technique) method is used to oversample the dataset
# SMOTE is used to balance the class distribution whenever it detects for an imbalance (one sample has signifcantly more samples than the other decreasing model performance) 
'''
Easy to understand example of SMOTE. Consider the following dataset with two features (X1 and X2) and a binary class label (y),
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |

There is an imbalance in the y column of the dataset as the class 1 is underrepresented.
If SMOTE is applied, the following output will be the result:
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |
| 2.75 | 4.25 |  1  |  # Synthetic sample (SMOTE)
| 4.25 | 2.75 |  1  |  # Synthetic sample (SMOTE)

As you can see, it generated two synthetic samples with the class 1 to balance out the class distribution in the dataset
'''

x_sample, y_sample = SMOTE().fit_resample(x, y.values.ravel())

x_sample = pd.DataFrame(x_sample)
y_sample = pd.DataFrame(y_sample)

# Seperate data into training and testing sets using the 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.2)

# Define a Sequential model
model = Sequential()

# Add Convolutional layers
# The first Conv1D layer with 32 filters, kernel size 3, and ReLU activation
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(16, 1)))
# Max pooling layer
model.add(MaxPooling1D(pool_size=2))

# Add another Conv1D layer with 16 filters, kernel size 3, and ReLU activation
model.add(Conv1D(16, kernel_size=3, activation='relu'))
# Max pooling layer
model.add(MaxPooling1D(pool_size=2))

# Flatten the output of the convolutional layers to pass it to the fully connected layers
model.add(Flatten())

# Add fully connected layers
# Dense layer with 8 neurons and ReLU activation
model.add(Dense(8, activation='relu'))

# Output layer with one neuron and Sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# Use Adam optimizer with a learning rate of 0.0001
# Use binary crossentropy as the loss function for binary classification
# Use accuracy as the evaluation metric
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define Callbacks
# ModelCheckpoint callback to save the best model during training based on validation loss
checkpoint_callback = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with callbacks
history = model.fit(x_train, y_train, epochs=10, batch_size=256, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), verbose=1)
# list all data in history
print(history.history.keys())

# TEST SUITE
pred_test = model.predict(x_test)
for i in range (len(pred_test)):
    if (pred_test[i] < 0.5):
        pred_test[i] = 0
    else:
        pred_test[i] = 1
pred_test = pred_test.astype(int)

def view_result(array):
    array = np.array(array)
    for i in range(len(array)):
        if array[i] == 0:
            print("Safe")
        else:
            print("Malicious")

print("PREDICTED RESULTS: ")
view_result(pred_test[:10])
print("\n")
print("ACTUAL RESULTS: ")
view_result(y_test[:10])


# SAVE MODEL
# model.save("Malicious_URL_Prediction.h5")
model.save(r'E:\BTECH\BTECH-3RD YEAR\SEM 06\B - ISM\Project_codes/Malicious_URL_Prediction.keras') 