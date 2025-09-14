import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# === GPU CONFIGURATION ===
print("[INFO] Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[INFO] GPU activated: {gpus[0].name}")
    except RuntimeError as e:
        print(f"[ERROR] Failed to set memory growth: {e}")
else:
    print("[WARNING] No GPU detected. Training will use CPU.")

# === CONFIG ===
DATASET_DIR = "processed_dataset"
EPOCHS = 20
BATCH_SIZE = 8
CLIP_LEN = 16
FRAME_SIZE = (112, 112)
NUM_CLASSES = 2  # 0 = NonFight, 1 = Fight

# === Load dataset with error handling ===
def load_data(split='train'):
    X, y = [], []
    class_names = ['Train_NonFight', 'Train_Fight']

    for label_index, label in enumerate(class_names):
        folder = os.path.join(DATASET_DIR, split, label)
        print(f"[INFO] Loading {label} from {folder}...")
        if not os.path.exists(folder):
            print(f"[WARNING] Folder does not exist: {folder}")
            continue
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                filepath = os.path.join(folder, file)
                try:
                    clip = np.load(filepath)
                    if clip.shape == (CLIP_LEN, *FRAME_SIZE, 3):
                        X.append(clip)
                        y.append(label_index)
                    else:
                        print(f"[WARNING] Skipped {file} due to shape mismatch: {clip.shape}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {filepath}: {e}")

    if len(X) == 0:
        raise RuntimeError("[FATAL] No valid .npy files loaded. Check your dataset.")

    print(f"[INFO] Loaded {len(X)} samples: {np.bincount(y)} (NonFight, Fight)")
    return np.array(X), np.array(y)

# === Load and prepare data ===
print("[INFO] Loading training data...")
X, y = load_data("train")

print("[INFO] Normalizing...")
X = X.astype("float32") / 255.0

# === Compute class weights for imbalance ===
print("[INFO] Computing class weights...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))
print(f"[INFO] Using class weights: {class_weight_dict}")

# === Define 3D CNN model ===
print("[INFO] Building model...")
model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation='relu', input_shape=(CLIP_LEN, *FRAME_SIZE, 3)),
    MaxPooling3D(pool_size=(2,2,2)),

    Conv3D(64, kernel_size=(3,3,3), activation='relu'),
    MaxPooling3D(pool_size=(2,2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train model ===
print("[INFO] Starting training...")
model.fit(
    X, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    class_weight=class_weight_dict
)

# === Save model ===
model.save("fight_detection_3dcnn.h5")
print("[INFO] Model saved to fight_detection_3dcnn.h5")
