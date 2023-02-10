import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

actions = [
          'Cross toe touch',
          'Knee touch',
          'Static lunge']

path_dir = 'C://Users/kwuser/Desktop/vscode/HAR/pose/kim-master/dataset'
folder_list = os.listdir(path_dir)

data = np.concatenate([
    np.load(path_dir + '/' + folder_list[3]),
    np.load(path_dir + '/' + folder_list[4]),
    np.load(path_dir + '/' + folder_list[5])
], axis=0)

x_data = data[:,:,:-1] #데이터에서 label값 분리
labels = data[:, 0, -1] #데이터에서 label값 지정

y_data = to_categorical(labels, num_classes=len(actions)) #lable -> onehot encoding

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2022)

x_train_sequence = x_train.shape[1]
x_val_sequence = x_val.shape[1]


scaler = StandardScaler()

for ss in range(x_train_sequence):
    scaler.partial_fit(x_train[:, ss, :])
    
result1=[]
for ss in range(x_train_sequence):
    result1.append(scaler.transform(x_train[:, ss, :]).reshape(x_train.shape[0], 1, x_train.shape[2]))
x_train_scaled=np.concatenate(result1, axis=1)
    
# for ss in range(x_val_sequence):
#     scaler.partial_fit(x_val[:, ss, :])

result2=[]
for ss in range(x_val_sequence):
    result2.append(scaler.transform(x_val[:, ss, :]).reshape(x_val.shape[0], 1, x_val.shape[2]))
x_val_scaled=np.concatenate(result2, axis=1)

model = Sequential([
    LSTM(64, activation='relu',return_sequences=True, input_shape=x_train_scaled.shape[1:3]),
    LSTM(128, activation='relu',return_sequences=True,),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(actions), activation='softmax')    
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(
    x_train_scaled,
    y_train,
    validation_data=(x_val_scaled, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('C://Users/kwuser/Desktop/vscode/HAR/pose/kim-master/models/scaled_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)
