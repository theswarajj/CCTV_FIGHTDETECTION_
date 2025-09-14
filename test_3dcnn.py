import os
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIG ===
MODEL_PATH = "fight_detection_3dcnn.h5"
VAL_DIR = "processed_dataset/val"
CLIP_LEN = 16
FRAME_SIZE = (112, 112)
LABELS = ['Train_NonFight', 'Train_Fight']

# === Load model ===
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# === Prediction loop ===
total = 0
correct = 0

label_map = {0: "Non-Fight", 1: "Fight"}

print("\n[INFO] Starting batch testing...\n")

for label_index, label_name in enumerate(LABELS):
    folder_path = os.path.join(VAL_DIR, label_name)
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            path = os.path.join(folder_path, file)
            clip = np.load(path)

            if clip.shape != (CLIP_LEN, *FRAME_SIZE, 3):
                print(f"[WARNING] Skipping {file}: Invalid shape {clip.shape}")
                continue

            clip = clip.astype("float32") / 255.0
            clip = np.expand_dims(clip, axis=0)

            pred = model.predict(clip, verbose=0)
            pred_label = np.argmax(pred)

            result = "✔️" if pred_label == label_index else "❌"
            if pred_label == label_index:
                correct += 1
            total += 1

            print(f"{result} {file} - Predicted: {label_map[pred_label]}, Actual: {label_map[label_index]}")

# === Summary ===
accuracy = correct / total if total > 0 else 0
print(f"\n[SUMMARY] Accuracy: {accuracy:.2%} ({correct}/{total})")
