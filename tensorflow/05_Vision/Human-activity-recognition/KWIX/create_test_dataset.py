import cv2
import numpy as np
import time
import os
from draw_function import set_cam, custom_landmarks
from operation import get_points_angle
from sklearn.preprocessing import Normalizer


actions = ['Babel squat', 'Dumbbell curl', 'Side lateral raise','Vent over low']

created_time = int(time.time())
seq_length = 15
N_scaler = Normalizer()

pose = set_cam()

path_dir = 'D://pose dataset/dataset_aihub_dumbbell/test(1000)' 
folder_list = os.listdir(path_dir)
os.makedirs('created_dataset', exist_ok=True) 


for idx, action in enumerate(actions):
    data = []
    action_dir = path_dir + '/' + folder_list[idx]
    
      
    for frame in os.listdir(action_dir):
        
        img = action_dir + '/' + frame
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)   
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks is not None:
            landmark_subset = custom_landmarks(results)
            joint, angle = get_points_angle(landmark_subset)

            reshape_angle = np.degrees(angle).reshape(-1, 1)
            scaled_angle = N_scaler.fit_transform(reshape_angle)

            angle_label = np.array([angle], dtype=np.float32)
            if idx == 3:
                label = 0
            else: 
                label = 1
            angle_label = np.append(angle_label, label)
            
            # angle_label = np.array([angle], dtype = np.float32) 
            # angle_label = np.append(angle_label, idx) #label 넣기

            d = np.concatenate([joint.flatten(), angle_label])
            data.append(d)
            
            #mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #landmark를 그리기

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    data = np.array(data)
    print(action, data.shape)
    np.save(os.path.join('C://Users/kwuser/Desktop/vscode/JH/kwix/HAR/dataset_test', f'raw_{action}_{created_time}'),data) #save raw data

    full_seq_data=[]
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])
    
    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join('C://Users/kwuser/Desktop/vscode/JH/kwix/HAR/dataset_test', f'seq_{action}_{created_time}'), full_seq_data) #save seq data    

