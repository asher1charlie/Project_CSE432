import os
import numpy as np
import pandas as pd
import librosa

df = pd.read_csv("data/metadata.csv")

rows = []

for i, row in df.iterrows():
    filepath = row["filepath"]

    # Load Audio
    y, sr = librosa.load(filepath, sr=48000)

    # Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y=y,sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # MFCC deltas
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Build feature dict or this file
    features = {}

    for j in range(13):
        features[f"mfcc_mean_{j+1}"] = mfcc_mean[j]
        features[f"mfcc_std_{j+1}"] = mfcc_std[j]
        features[f"mfcc_delta_mean{j+1}"] = mfcc_delta_mean[j]
        features[f"mfcc_delta_std_{j+1}"] = mfcc_delta_std[j]

    for j in range(12):
        features[f"chroma_mean_{j+1}"] = chroma_mean[j]
        features[f"chroma_std_{j+1}"] = chroma_std[j]

    features["zcr_mean"]        = zcr_mean
    features["zcr_zcr"]         = zcr_std
    features["rms_mean"]        = rms_mean
    features["rms_std"]         = rms_std
    features["centroid_mean"]   = centroid_mean
    features["centroid_std"]    = centroid_std
    features["bandwidth_mean"]  = bandwidth_mean
    features["bandwidth_std"]   = bandwidth_std
    features["rolloff_mean"]    = rolloff_mean
    features["rolloff_std"]     = rolloff_std

    # Add labes
    features["emotion"]         = row["emotion"]
    features["emotion_id"]      = row["emotion_id"]
    features["vocal_channel"]   = row["vocal_channel"]
    features["actos"]           = row["actor"]
    features["gender"]          = row["gender"]

    rows.append(features)

    # Progressupdate every 100 files
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(df)} files")

features_df = pd.DataFrame(rows)
features_df.to_csv("data/features.csv", index=False)
print(f"\nSaved {len(features_df)} rows to data/features.csv")
print(f"Feature vector length: {len(features_df.columns)} columns")
print(features_df.head())