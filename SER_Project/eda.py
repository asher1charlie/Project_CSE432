import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# - **Source:** [https://zenodo.org/records /1188976](https://zenodo.org/records/1188976)
# - **Citation (required in your report):** Livingstone SR, Russo FA (2018).
#  _The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, 
# multimodal set of facial and vocal expressions in North American English._ PLoS ONE 13(5): e0196391.
# [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)


# Label Maps
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

rows = []


for root, dirs, files in os.walk("data"):
    for filename in files:
        if not filename.endswith(".wav"):
            continue

        parts = filename.replace(".wav", "").split("-")
        if len(parts) != 7:
            continue

        modality = parts[0]
        if modality != "03": # audio-files only
            continue

        vocal = "speech" if parts[1] == "01" else "song"
        emotion = EMOTION_MAP[parts[2]]
        intensity = "normal" if parts[3] == "01" else "strong"
        actor = int(parts[6])
        gender = "male" if actor % 2 == 1 else "female"

        rows.append({
            "filename": filename,
            "filepath": os.path.join(root, filename),
            "vocal_channel": vocal,
            "emotion": emotion,
            "emotion_id": int(parts[2]),
            "intensity": intensity,
            "actor": actor,
            "gender": gender,
        })

df = pd.DataFrame(rows)
print(f"Total files: {len(df)}")
print(df.head())

# save to csv
df.to_csv("data/metadata.csv", index=False)
print("Saved to metadata")


# Emotion counts
plt.figure(figsize=(10, 5))
df["emotion"].value_counts().plot(kind="bar")
plt.title("Files per emotion")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.savefig("data/emotion_counts.png")
plt.show()


# Emotion counts vocal channel (speechh vs song)
speech_counts = df[df["vocal_channel"] == "speech"]["emotion"].value_counts()
song_counts = df[df["vocal_channel"] == "song"]["emotion"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
speech_counts.plot(kind="bar", ax=axes[0], title="Speech")
song_counts.plot(kind="bar", ax=axes[1], title="Song")
for ax in axes:
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("data/speech_vs_song.png")
plt.show()

# Gender Breakdown
plt.figure(figsize=(10, 5))
df.groupby(["emotion", "gender"]).size().unstack().plot(kind="bar")
plt.title("Emotion counts by gender")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("data/gender_emotion.png")
plt.show()
