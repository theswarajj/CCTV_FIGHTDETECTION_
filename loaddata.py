def load_data(split='train'):
    X, y = []
    class_names = ['Train_NonFight', 'Train_Fight']
    
    for label_index, label in enumerate(class_names):
        folder = os.path.join(DATASET_DIR, split, label)
        print(f"[INFO] Loading {label} from {folder}...")
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                filepath = os.path.join(folder, file)
                try:
                    print(f"[DEBUG] Loading: {filepath}")
                    clip = np.load(filepath)
                    if clip.shape == (CLIP_LEN, *FRAME_SIZE, 3):
                        X.append(clip)
                        y.append(label_index)
                    else:
                        print(f"[WARNING] Skipped {file}, invalid shape: {clip.shape}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {filepath}: {e}")
                    
    return np.array(X), np.array(y)
