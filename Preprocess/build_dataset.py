import os
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from scipy.fft import fft


counter = 0
CLIP_LENGTH = 32

def process_folder(folder, label):

    global counter

    if not os.path.exists(folder):
        print(f"Error: Directory not found: {folder}")
        return

    videos = os.listdir(folder)

    for video in tqdm(videos):

        path = os.path.join(
            folder,
            video
        )

        frames = extract_faces(path) # Removed batch_size argument

        if len(frames) < CLIP_LENGTH:
            continue

        for i in range(
            0,
            len(frames)-CLIP_LENGTH,
            CLIP_LENGTH
        ):

            clip = frames[
                i:i+CLIP_LENGTH
            ]

            pwtf = generate_pwtf(
                clip
            )

            save_path = os.path.join(
                SAVE_DIR,
                f"{counter}_{label}.npy"
            )

            np.save(
                save_path,
                pwtf
            )

            counter += 1

    #process real videos
    process_folder(
    REAL_DIR,
    0)

    #process deepfake videos
    process_folder(
    FAKE_DIR,
    1)

