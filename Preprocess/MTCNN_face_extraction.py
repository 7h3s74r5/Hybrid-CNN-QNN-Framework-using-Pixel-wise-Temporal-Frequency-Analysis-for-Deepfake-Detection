import cv2
import numpy as np

from facenet_pytorch import MTCNN
from tqdm import tqdm # Import tqdm for the new loop


detector = MTCNN(
    image_size=128,
    margin=10,
    keep_all=False,
    post_process=False,
    device=device
)

def extract_faces(
    video_path,
    frame_skip=4
):
    """
    Returns:
        np.ndarray
        shape -> (N,128,128)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return np.array([])

    rgb_frames = []

    frame_idx = 0
    total_frames_read = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames_read += 1

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frames.append(rgb)

        frame_idx += 1

    cap.release()

    print(f"Total frames read from video: {total_frames_read}")
    print(f"Frames to process after skipping: {len(rgb_frames)}")

    if len(rgb_frames) == 0:
        print("No frames collected after skipping. Check video or frame_skip.")
        return np.array([])

    faces = []
    detected_count = 0
    no_face_count = 0

    for i, rgb in enumerate(tqdm(rgb_frames, desc="Detecting faces")):
        try:
            face = detector(rgb)
            if face is None:
                no_face_count += 1
                continue
            face = face.cpu().numpy()
            face = np.transpose(
                face,
                (1,2,0)
            )
            gray = cv2.cvtColor(
                face,
                cv2.COLOR_RGB2GRAY
            )
            faces.append(gray)
            detected_count += 1
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    print(f"Faces detected in {detected_count} frames.")
    print(f"No faces found in {no_face_count} frames.")

    return np.array(
        faces,
        dtype=np.float32
    )
