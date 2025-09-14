import os
import cv2 
import numpy as np
from tqdm import tqdm

VIDEO_DIR = "RWF-2000"
OUTPUT_DIR = "processed_dataset"
FRAME_SIZE = (112, 112)
CLIP_LEN = 16

def extract_clips_from_video(video_path, clip_len=CLIP_LEN):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    clips = []
    for i in range(0, len(frames) - clip_len + 1, clip_len):
        clip = frames[i:i+clip_len]
        clips.append(np.array(clip))

    return clips

def process_directory(split):
    for label in ["Train_Fight", "Train_NonFight"]:
        input_path = os.path.join(VIDEO_DIR, split, label)
        output_path = os.path.join(OUTPUT_DIR, split, label.lower())

        if not os.path.exists(input_path):
            print(f"[ERROR] Input path not found: {input_path}")
            continue

        os.makedirs(output_path, exist_ok=True)

        video_files = [f for f in os.listdir(input_path) if f.endswith((".mp4", ".avi"))]

        clip_counter = 0
        for video in tqdm(video_files, desc=f"Processing {split}/{label}"):
            video_path = os.path.join(input_path, video)
            clips = extract_clips_from_video(video_path)

            for clip in clips:
                if clip.size == 0:
                    print(f"[WARNING] Empty clip at {video_path}, skipping...")
                    continue

                np.save(os.path.join(output_path, f"clip_{clip_counter:05d}.npy"), clip)
                clip_counter += 1

for split in ["train", "val"]:
    process_directory(split)