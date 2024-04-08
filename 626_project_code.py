# -*- coding: utf-8 -*-
"""
Biostat 626: Project Code
@author: Ethan W, Will T
"""
# this is a demo code for computing the average intensity of proteins of all images (save as .csv)
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------

# Average intensity of training data
img_dir = 'train/images' # dir that saves your images
n_train = 225 # number of images
n_protein = 52 # number of proteins 

avg_list = [] 
for i in tqdm(np.arange(1, n_train+1), total=n_train, desc="Processing"):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)
    avg = np.mean(img, axis=(1,2))
    avg_list.append([i] + avg.tolist())
avg_df = pd.DataFrame(avg_list, columns = ['id'] + ['protein' + str(i) for i in range(1, 52+1)])    

# Survival time data
osmonth_df = pd.read_csv('train/train_data.csv')

# Testing data average intensity
test_dir = 'test/images'
n_test = 56

test_list = [] 
for i in tqdm(np.arange(226, 226+n_test), total=n_test, desc="Processing"):
    img_file_name = f'{i}.tiff'
    path = os.path.join(test_dir, img_file_name)
    img = imageio.v2.imread(path)
    avg = np.mean(img, axis=(1,2))
    test_list.append([i] + avg.tolist())
avgtest_df = pd.DataFrame(test_list, columns = ['id'] + ['protein' + str(i) for i in range(1, 52+1)])    

# ----------------------------------------------------------------------------

# Training data
X_train = avg_df 
y_train = osmonth_df
# testing data
X_test = avgtest_df
X_test = X_test.drop("id", axis = 1)

# create histogram of training data
plt.figure(figsize=(10,6))
plt.hist(y_train, bins = 25, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Survival Time (training set)', fontsize = 16)
plt.xlabel('Survival Time (months)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# create correlation matrix of training data
f = plt.figure(figsize=(19,15))
plt.matshow(X_train.corr(), fignum=f.number)
plt.xticks(range(X_train.select_dtypes(['number']).shape[1]), X_train.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.select_dtypes(['number']).shape[1]), X_train.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

# ----------------------------------------------------------------------------

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=n_protein, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
history

# Model evaluations
loss, mae = model.evaluate(X_train, y_train)
print("Mean Absolute Error:", mae)

# Predictions
predictions = model.predict(X_test)


# Create histogram (testing data)
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Survival Time (testing set)', fontsize=16)
plt.xlabel('Survival Time (months)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Hyperparameter tuning/grid search
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=52, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

# Wrap Keras model so it can be used by scikit-learn
model = KerasRegressor(build_fn=create_model, verbose=0)

param_grid = {
    'epochs': [50, 100, 150],  # Number of epochs to train the model
    'batch_size': [32, 64, 128]  # Batch size used during training
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))