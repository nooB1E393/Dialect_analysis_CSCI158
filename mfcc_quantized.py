from sklearn.cluster import KMeans
import numpy as np
import os

def quantize_features(features, centroids):
    """Quantize the features based on centroids."""
    quantized_features = []
    for feature in features:
        distances = np.linalg.norm(centroids - feature, axis=1)
        nearest_centroid_idx = np.argmin(distances)
        quantized_features.append(nearest_centroid_idx)
    return quantized_features

# Define paths
input_folder_path = 'mfccFeat_norm/South_mfcc'  # Folder with original .npy MFCC feature files
output_folder_path = 'quantized_mfcc_features/South_mfccq'  # Folder to store quantized features

# Ensure the output folder exists
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Load MFCC features
mfcc_feature_files = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith('.npy')]
mfcc_features = [np.load(f) for f in mfcc_feature_files]

# Prepare data for K-means (flatten all features into a single array)
mfcc_all = np.vstack([features.T for features in mfcc_features])  # Each column is a feature vector

# Train K-means
n_clusters = 10  # Define the number of clusters (codebook size)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mfcc_all)

# Quantize features using the trained centroids
for i, mfcc in enumerate(mfcc_features):
    quantized = quantize_features(mfcc.T, kmeans.cluster_centers_)
    output_file_path = os.path.join(output_folder_path, os.path.basename(mfcc_feature_files[i]).replace('.npy', '_quantized.npy'))
    np.save(output_file_path, quantized)

print("Quantization complete.")
