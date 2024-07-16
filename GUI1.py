import os
import librosa
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# 加载保存的模型
clf = joblib.load('model1.pkl')

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

# 预测函数
def predict_audio(file_path):
    feature = extract_features(file_path)
    if feature is not None:
        feature_df = pd.DataFrame([feature])
        prediction = clf.predict(feature_df)
        return "OK" if prediction[0] == 0 else "NG"
    else:
        return "Error"

# 选择文件并预测
def select_file_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav")])
    if file_path:
        result = predict_audio(file_path)
        messagebox.showinfo("Prediction Result", f"The prediction for the audio file is: {result}")

# 创建主窗口
root = tk.Tk()
root.title("Audio Classification")

# 设置主窗口的大小
root.geometry("600x400")  # 设置窗口的宽度和高度，例如 600x400

# 创建按钮
select_button = tk.Button(root, text="Select Audio File", command=select_file_and_predict)
select_button.pack(pady=20)

# 运行主循环
root.mainloop()