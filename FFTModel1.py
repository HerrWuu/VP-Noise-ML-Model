import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# 读取音频文件
sampling_rate, audio_data = wav.read('c:/Users/herrw/Desktop/NG1.wav')

# 处理音频数据格式问题
if audio_data.dtype != np.float32:
    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

# 计算FFT
fft_result = fft(audio_data)

# 获取频谱的幅度
magnitude = np.abs(fft_result)

# 计算频率轴
freqs = np.linspace(0, sampling_rate, len(magnitude))

# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])  # 只绘制一半的频谱
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum of Audio Signal')
plt.yscale('log')  # 使用对数刻度以便更好地显示频谱
plt.grid(True)
plt.show()