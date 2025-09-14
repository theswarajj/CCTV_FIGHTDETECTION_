import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class VideoClipDataset(Dataset):
    def __init__(self, root_dir, clip_len=16):
        self.samples = []
        self.clip_len = clip_len

        for label, folder_name in enumerate(['NonFight', 'Fight']):
            full_path = os.path.join(root_dir, folder_name)
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith(".mp4"):
                        video_path = os.path.join(root, file)
                        self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()

        if len(frames) < self.clip_len:
            frames += [frames[-1]] * (self.clip_len - len(frames))

        frames = frames[:self.clip_len]
        clip = np.stack(frames, axis=0)  # (D, H, W, C)
        clip = clip.transpose(3, 0, 1, 2)  # (C, D, H, W)
        return torch.tensor(clip, dtype=torch.float32) / 255.0, torch.tensor(label)

