import librosa
import numpy as np
print("NumPy version:", np.__version__)
import matplotlib.pyplot as plt

# 加载音频文件
audio_path = r'C:\\Users\\herrw\\Desktop\\VP-Noise-ML-Model\\音频文件\\NG\\NG1.wav'
y, sr = librosa.load(audio_path, sr=None)

# 计算 FFT
fft_result = np.fft.fft(y)
freqs = np.fft.fftfreq(len(y), 1/sr)

# 取频谱的幅度部分
magnitude = np.abs(fft_result)

# 绘制频谱图
plt.figure(figsize=(12, 6))
plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Audio Spectrum')
plt.xlim(0, sr/2)  # 设置 x 轴范围，只显示正频率部分
plt.show()