import os
import pandas as pd 
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import time

# Data Analyiss
train =  pd.read_csv('tensorflow/Linear_Regression/Input/train.csv')
test = pd.read_csv('tensorflow/Linear_Regression/Input/test.csv')

print(train.head())

# Data Preprocessing

# Drop null values
train = train.dropna()

# Set training data and targets
X_train = train['x']
y_train = train['y']

# Set testing data and targets
X_test = test['x']
y_test = test['y']

# Careful: We need to reshape the data in order to fit our model
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1,1)

# Now let's scale the data
# Scaling the data helps our model converge faster
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Visualize the Data
#px.scatter(x=train['x'], y=train['y'],template='gridon')

# Model
model = LinearRegression() #Create linear regression instance
start = time.time()
model.fit(X_train, y_train)
end = time.time()

# Evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions) #Get the mean squared error as the evaluation metric
print(f'the mean squared error is: {mse}')
print("learning time: ", end - start)
