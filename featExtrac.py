import librosa
import numpy as np
import os

folder_path = './audio/new'  # Your specified folder path
output_folder_path = './mfcc_features'  # Folder to store MFCC features

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)


def extract_mfcc(file_path, num_mfcc=15, n_fft=2048, hop_length=512):
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs


# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav'):
        file_path = os.path.join(folder_path, file_name)

        # Extract MFCC features
        mfccs = extract_mfcc(file_path)

        # Define the output path for the MFCC features
        output_file_path = os.path.join(output_folder_path, file_name.replace('.wav', '.npy'))

        # Save the MFCC features to a .npy file
        np.save(output_file_path, mfccs)

print("MFCC feature extraction complete.")
