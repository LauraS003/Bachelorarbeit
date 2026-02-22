import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VIDEO_PATH = r"E:\BA\Messungen\camera_front_tele_30fov (1).mp4"

USE_ROI = False
ROI = (0, 0, 0, 0)  

 


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))


def laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def maybe_crop_roi(frame_bgr: np.ndarray, use_roi: bool, roi: tuple[int, int, int, int]) -> np.ndarray:
    if not use_roi:
        return frame_bgr
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("ROI w/h müssen > 0 sein, wenn USE_ROI=True.")
    return frame_bgr[y:y + h, x:x + w]


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # Fallback: wenn FPS nicht lesbar ist
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = maybe_crop_roi(frame, USE_ROI, ROI)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = frame_idx / fps
        ten = tenengrad(gray)
        lapv = laplacian_variance(gray)

        rows.append({
            "frame": frame_idx,
            "time_s": t,
            "tenengrad": ten,
            "laplacian_variance": lapv,
        })

        frame_idx += 1

        if total_frames and frame_idx % 100 == 0:
            print(f"{frame_idx}/{total_frames} Frames verarbeitet...")
        elif frame_idx % 200 == 0:
            print(f"{frame_idx} Frames verarbeitet...")

    cap.release()

    if not rows:
        raise RuntimeError("Keine Frames gelesen. Ist das Video leer/defekt?")

    df = pd.DataFrame(rows)

    out_dir = video_path.parent
    base = video_path.stem

    csv_path = out_dir / f"{base}_sharpness.csv"
    png_path = out_dir / f"{base}_sharpness.png"

    df.to_csv(csv_path, index=False, sep=";")  

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["time_s"], df["tenengrad"], label="Tenengrad")
    plt.plot(df["time_s"], df["laplacian_variance"], label="Laplacian-Varianz")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Schärfemetrik (a.u.)")
    plt.title(f"Schärfe über Zeit – {video_path.name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print("Fertig ✅")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()
