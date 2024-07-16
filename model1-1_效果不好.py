import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# 定义文件夹路径
normal_folder = r'C:\\Users\\herrw\\Desktop\\VP-Noise-ML-Model\\音频文件\\OK'
abnormal_folder = r'C:\\Users\\herrw\\Desktop\\VP-Noise-ML-Model\\音频文件\\NG'

# 提取特征的函数
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# 读取文件并提取特征
def load_data(folder, label):
    features = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if file_path.endswith('.wav'):
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
    labels = [label] * len(features)
    return features, labels

# 加载正常和异常数据
normal_features, normal_labels = load_data(normal_folder, 0)
abnormal_features, abnormal_labels = load_data(abnormal_folder, 1)

# 合并数据
features = normal_features + abnormal_features
labels = normal_labels + abnormal_labels

# 创建 DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# 分割数据集
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建随机森林分类器
clf = RandomForestClassifier(random_state=42)

# 参数调优
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 最佳参数
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 使用最佳参数训练模型
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 保存模型
joblib.dump(clf, 'random_forest_model.pkl')