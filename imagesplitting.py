import os
import shutil
import pandas as pd

# Read train.csv file
df = pd.read_csv('D:/College/Semester 7/Project/Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow/train_data.csv')

# Create two folders to store images with 0 and 1 prediction
os.makedirs('Data/No Pnuemothorax', exist_ok=True)
os.makedirs('Data/Pnuemothorax', exist_ok=True)

# Iterate over each row of the dataframe and move the corresponding image to the corresponding folder
for index, row in df.iterrows():
    file_name = row['file_name']
    target = row['target']
    
    source_path = os.path.join('D:/College/Semester 7/Project/Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow/Dataset', file_name)
    
    if target == 0:
        destination_path = os.path.join('D:/College/Semester 7/Project/Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow/Data/No Pnuemothorax', file_name)
    elif target == 1:
        destination_path = os.path.join('D:/College/Semester 7/Project/Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow/Data/Pnuemothorax', file_name)
    
    shutil.move(source_path, destination_path)
