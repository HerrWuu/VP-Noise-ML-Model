import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# file path
normal_folder = r'C:\\Users\\herrw\\Desktop\\VP-Noise-ML-Model\\NoiseData\\OK'      #change to ur own path
abnormal_folder = r'C:\\Users\\herrw\\Desktop\\VP-Noise-ML-Model\\NoiseData\\NG'

# extract feature function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)
    return mfccs_mean

# load and extract feature
def load_data(folder, label):
    features = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if file_path.endswith('.wav'):
            feature = extract_features(file_path)
            features.append(feature)
    labels = [label] * len(features)
    return features, labels

# load data
normal_features, normal_labels = load_data(normal_folder, 0)
abnormal_features, abnormal_labels = load_data(abnormal_folder, 1)

# conbine data 
features = normal_features + abnormal_features
labels = normal_labels + abnormal_labels

# build DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# divide dataset
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# train
clf.fit(X_train, y_train)

# prediction
y_pred = clf.predict(X_test)

# calculating accurancy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

joblib.dump(clf, 'model1.pkl')