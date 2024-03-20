import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

folder_path = './audio/new'  # Your specified folder path

# Initialize a figure for plotting
plt.figure(figsize=(14, 8))

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):
        file_path = os.path.join(folder_path, file_name)

        # Load the audio file
        signal, sr = librosa.load(file_path, sr=None)

        # Apply FFT
        fft_output = np.fft.fft(signal)

        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(fft_output)

        # Compute frequency bins
        frequency_bins = np.linspace(0, sr, len(magnitude_spectrum))

        # Plot the magnitude spectrum
        plt.plot(frequency_bins[:len(magnitude_spectrum) // 2], magnitude_spectrum[:len(magnitude_spectrum) // 2],
                 label=os.path.basename(file_path))

# Configure the plot
plt.title('Frequency Spectrum Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.02))  # Adjust legend position
plt.tight_layout()

# Show the plot
plt.show()
