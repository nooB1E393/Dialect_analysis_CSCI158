

import os
import librosa
import soundfile as sf

def trim_or_pad_audio_file(input_file_path, outfile_path, target_duration):
    # Load the audio file
    audio, sr = librosa.load(input_file_path, sr=None)
    # Calculate the target length
    target_length = int(target_duration * sr)

    if len(audio) >= target_length:
        if len(audio) > target_length:
            trimmed_audio = audio[:target_length]
        sf.write(outfile_path, trimmed_audio, sr)
    


   


dataset_directory = 'EngMen/Irish'
output_directory = '158/new_Irish'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(dataset_directory):
    if filename.endswith(".wav"):
        input_file_path = os.path.join(dataset_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        trim_or_pad_audio_file(input_file_path, output_file_path, 4)
        print(f"Processed {filename}")
