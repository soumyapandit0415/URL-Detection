# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import sklearn.metrics as metrics
from scipy.stats import randint, uniform
import pickle
from xgboost import XGBClassifier


#Data Pre-Processing
urldata = pd.read_csv("./Url_Processed.csv")
# droping "Unnamed: 0" as its unncessary feature
urldata.drop("Unnamed: 0",axis=1,inplace=True)
# remove uneccessary columns
urldata.drop(["url","label"],axis=1,inplace=True)
#Printing the 1st 5 rows of the dataset
# print(urldata.head())
#Independent Variables
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]
#Dependent Variable
y = urldata['result']
# printing x
print(x.head())
#printing y
print(y.head())

#Oversampling using SMOTE
from imblearn.over_sampling import SMOTE
x_sample, y_sample = SMOTE().fit_resample(x, y.values.ravel())
x_sample = pd.DataFrame(x_sample)
y_sample = pd.DataFrame(y_sample)

# checking the sizes of the sample data
print("\nSample data Size:\n")
print("Size of x-sample :", x_sample.shape)
print("Size of y-sample :", y_sample.shape)
# DATA SPLITTING 
print("\nData Splitting\n")
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.2)
print("Shape of x_train: ", x_train.shape)
print("Shape of x_valid: ", x_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_valid: ", y_test.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

#Creating a dictionary for each model's report
classification_reports = {}


''' RANDOM FOREST ''' 

print("\n\nRANDOM FOREST\n")
# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(max_depth=7)
rf.fit(x_train_scaled, np.ravel(y_train))  # Convert y_train to 1D array
# Predict on the test set
rf_predict = rf.predict(x_test_scaled)
# Calculate other metrics
rf_Accuracy_Score = accuracy_score(y_test, rf_predict)
rf_JaccardIndex = jaccard_score(y_test, rf_predict)
rf_F1_Score = f1_score(y_test, rf_predict)
rf_Log_Loss = log_loss(y_test, rf_predict)
print(f"Accuracy: {rf_Accuracy_Score}")
print(f"Jaccard Index: {rf_JaccardIndex}")
print(f"F1 Score: {rf_F1_Score}")
print(f"Log Loss: {rf_Log_Loss}")
# Generate confusion matrix
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
# Plot confusion matrix
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of RANDOM FOREST')
plt.show()
# calculating other classification metrics
# Calculate classification report for Random Forest
rf_report = classification_report(y_test, rf_predict, target_names=["legitimate", "malicious"])
classification_reports["Random Forest"] = rf_report

'''SUPPORT VECTOR MACHINE '''
'''
print("\n\nSUPPORT VECTOR MACHINE\n")
svm = SVC()
#Model Training
svm.fit(x_train_scaled, np.ravel(y_train))
#Model Test
svm_predict = svm.predict(x_test_scaled)
#Evaluation Metrics
svm_Accuracy_Score = accuracy_score(y_test,svm_predict)
svm_JaccardIndex = jaccard_score(y_test,svm_predict)
svm_F1_Score = f1_score(y_test,svm_predict)
svm_Log_Loss = log_loss(y_test,svm_predict)
#Printing the evaluation metrics
print(f"Accuracy: {svm_Accuracy_Score}")
print(f"Jaccard Index: {svm_JaccardIndex}")
print(f"F1 Score: {svm_F1_Score}")
print(f"Log Loss: {svm_Log_Loss}")
#Confusion Matrix
svm_conf_matrix = confusion_matrix(y_test,svm_predict)
# Plot confusion matrix
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of SVM')
plt.show()
# Calculate classification report for SVM
svm_report = classification_report(y_test, svm_predict, target_names=["legitimate", "malicious"])
classification_reports["Support Vector Machine"] = svm_report
'''

'''XGBOOST'''

print("\n\nXGBOOST\n")
params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'objective': 'binary:logistic'
}
#Model initialization and training
xgb = XGBClassifier(**params)
xgb.fit(x_train_scaled,y_train)
#Model testing
xgb_predict = xgb.predict(x_test_scaled)
#Evaluation Metrics Calculation
xgb_Accuracy_Score = accuracy_score(y_test,xgb_predict)
xgb_JaccardIndex = jaccard_score(y_test,xgb_predict)
xgb_F1_Score = f1_score(y_test,xgb_predict)
xgb_Log_Loss = log_loss(y_test,xgb_predict)
#Printing Evaluation Metrics
print(f"Accuracy: {xgb_Accuracy_Score}")
print(f"Jaccard Index: {xgb_JaccardIndex}")
print(f"F1 Score: {xgb_F1_Score}")
print(f"Log Loss: {xgb_Log_Loss}")
#Calculating Confusion matrix & plotting
xgb_conf_matrix = confusion_matrix(y_test,xgb_predict)
sns.heatmap(xgb_conf_matrix,annot=True, fmt = 'd',cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of XGBOOST')
plt.show()
# Calculate classification report for XGBoost
xgb_report = classification_report(y_test, xgb_predict, target_names=["legitimate", "malicious"])
classification_reports["XGBoost"] = xgb_report


''' MULTI - LAYER PERCEPTRON ''' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization ,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

print("\n\nMulti Layer Perceptron\n")
# model building
model_mlp = Sequential()
model_mlp.add(Dense(32, activation = 'relu', input_shape = (16, )))
model_mlp.add(Dense(16, activation='relu'))
model_mlp.add(Dense(8, activation='relu')) 
model_mlp.add(Dense(1, activation='sigmoid'))
model_mlp.summary()
# compiling model
opt = keras.optimizers.Adam(lr=0.0001)
model_mlp.compile(optimizer= opt ,loss='binary_crossentropy',metrics=['acc'])

# custom callback to stop the training when certain metric value is reached

# stop training when validation loss reach 0.1
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<0.1):
            print("\nReached 0.1 val_loss so cancelling training!")
            self.model.stop_training = True
        
callback = myCallback()

# start training the model 
history = model_mlp.fit(x_train, y_train, epochs=10,batch_size=256, callbacks=[callback],validation_data=(x_test,y_test),verbose=1)

# DISPLAYING MODEL TRAINING HISTORY

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.figure(figsize=(20,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predicting on test data.
pred_test = model_mlp.predict(x_test)
# pred_test = model.predict(test_data)
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
            print("Non Mallicious")
        else:
            print("Mallicious")

print("PREDICTED : ")
view_result(pred_test[:10])
print("\n")
print("ACTUAL : ")
view_result(y_test[:10])

# Convert predicted probabilities to binary predictions
binary_pred_test = (pred_test > 0.5).astype(int)
# Generate classification report
# Calculate classification report for MLP
mlp_report = classification_report(y_test, binary_pred_test, target_names=["legitimate", "malicious"])
classification_reports["Multi-Layer Perceptron"] = mlp_report



# CNN
print("\n\nConvolutional Neural Networks:\n")
def CNN(input_shape):

    model = keras.Sequential()
    model.add(layers.Input(input_shape))  # Here input_shape should be the shape of the input data
    model.add(layers.Conv1D(filters = 16,kernel_size = 3,activation = 'relu',padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
    model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = 'relu',padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
    model.add(layers.Conv1D(filters = 64,kernel_size = 3,activation = 'relu',padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
    model.add(layers.Conv1D(filters = 128,kernel_size = 3,activation = 'relu',padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
    model.add(layers.Conv1D(filters = 256,kernel_size = 3,activation = 'relu',padding = 'same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size = 2,padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    
    return model

# Reshape input data to include sequence_length dimension
x_train_reshaped = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_reshaped = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)
# Get the shape of reshaped input data
input_shape = x_train_reshaped.shape[1:]

# CNN model
CNN_model1 = CNN(input_shape)
print(CNN_model1.summary())
# compiling model
opt = keras.optimizers.Adam(lr=0.0001)
CNN_model1.compile(optimizer= opt ,loss='binary_crossentropy',metrics=['acc'])

# custom callback to stop the training when certain metric value is reached

# stop training when validation loss reach 0.1
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<0.1):
            print("\nReached 0.1 val_loss so cancelling training!")
            self.model.stop_training = True
        
callback = myCallback()

# start training the model 
history = CNN_model1.fit(x_train_reshaped, y_train, epochs=10,batch_size=256, callbacks=[callback],validation_data=(x_test_reshaped,y_test),verbose=1)

# DISPLAYING MODEL TRAINING HISTORY

# list all data in history
print(history.history.keys())

pred_test = CNN_model1 .predict(x_test_reshaped)
# pred_test = model.predict(test_data)
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
            print("Non Mallicious")
        else:
            print("Mallicious")
# Convert predicted probabilities to binary predictions
binary_pred_test = (pred_test > 0.5).astype(int)
# Calculate classification report for CNN
cnn_report = classification_report(y_test, binary_pred_test, target_names=["legitimate", "malicious"])
classification_reports["Convolutional Neural Networks"] = cnn_report
print("PREDICTED : ")
view_result(pred_test[:10])
print("\n")
print("ACTUAL : ")
view_result(y_test[:10])
# Assuming cnn_model is the best-performing CNN model
CNN_model1.save('E:/BTECH/BTECH-3RD YEAR/SEM 06/B - ISM/Project_codes/Website/best_cnn_model.h5')
# Print classification reports for each model
for model_name, report in classification_reports.items():
    print(f"{model_name} Report:\n{report}\n")