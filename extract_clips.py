import os
import cv2

# Parameters
clip_len = 16  # Number of frames per clip
resize_dim = (112, 112)

def save_clip(frames, out_path):
    """Save a list of frames as an .mp4 video clip."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 5.0, resize_dim)

    for frame in frames:
        out.write(frame)
    out.release()

def extract_clips_from_folder(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    # Recursively search for all .mp4 files
    video_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    print(f"[INFO] Found {len(video_files)} videos in {src_dir}")
    clip_count = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize_dim)
            frames.append(frame)

            if len(frames) == clip_len:
                out_path = os.path.join(dst_dir, f"clip_{clip_count}.mp4")
                save_clip(frames, out_path)
                clip_count += 1
                frames = frames[4:]  # Use sliding window (optional)

        cap.release()

    print(f"[DONE] Saved {clip_count} clips to {dst_dir}")

# Main dataset conversion
if __name__ == "__main__":
    dataset_map = {
        "RWF-2000/train/Train_Fight": "processed_dataset/train/Fight",
        "RWF-2000/train/Train_NonFight": "processed_dataset/train/NonFight",
        "RWF-2000/val/val_Fight": "processed_dataset/val/Fight",
        "RWF-2000/val/val_NonFight": "processed_dataset/val/NonFight",
    }

    for src, dst in dataset_map.items():
        extract_clips_from_folder(src, dst)
