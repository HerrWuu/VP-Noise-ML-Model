import librosa
import numpy as np
print("NumPy version:", np.__version__)
import matplotlib.pyplot as plt

audio_path = r''
y, sr = librosa.load(audio_path, sr=None)

fft_result = np.fft.fft(y)
freqs = np.fft.fftfreq(len(y), 1/sr)

magnitude = np.abs(fft_result)
plt.figure(figsize=(12, 6))
plt.plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Audio Spectrum')
plt.xlim(0, sr/2)  
plt.show()