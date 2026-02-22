from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



CAM2_DIR = Path(r"L:\BA\Messungen\Frequenzen\Videos\Cam2")  
OUT_DIR  = CAM2_DIR / "_Auswertung_OpticalFlow_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_CENTER_ROI = True
ROI_FRAC = 0.70

PAIR_STRIDE = 1     
MAX_PAIRS = None     

# Farnebäck Optical Flow
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=0
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

def export_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Plot: {path}")

def get_center_roi(gray: np.ndarray, frac: float) -> np.ndarray:
    h, w = gray.shape
    frac = float(np.clip(frac, 0.05, 1.0))
    rh = int(h * frac)
    rw = int(w * frac)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    return gray[y0:y0 + rh, x0:x0 + rw]

def parse_frequency_from_folder(folder_name: str) -> int | None:
    if folder_name.lower() == "ruhe":
        return 0
    m = re.match(r"^\s*(\d+)\s*hz\s*$", folder_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_signal_from_filename(fname: str) -> str | None:
    low = fname.lower()
    if "sinus" in low:
        return "Sinus"
    if "rechteck" in low:
        return "Rechteck"
    if "ruhe" in low:
        return "Ruhe"
    return None

def compute_flow_timeseries(video_path: Path) -> tuple[pd.DataFrame, float]:
   
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float("nan")

    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Video hat keine Frames: {video_path}")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if USE_CENTER_ROI:
        prev_gray = get_center_roi(prev_gray, ROI_FRAC)

    records = []
    pair_index = 0
    kept = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pair_index += 1
        if PAIR_STRIDE > 1 and (pair_index - 1) % PAIR_STRIDE != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if USE_CENTER_ROI:
            gray = get_center_roi(gray, ROI_FRAC)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FB_PARAMS)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        mag_median = float(np.median(mag))
        mag_mean = float(np.mean(mag))
        mag_p95 = float(np.quantile(mag, 0.95))

        if not np.isnan(fps) and fps > 0:
            t_s = (pair_index * PAIR_STRIDE) / fps
        else:
            t_s = float(pair_index)

        records.append({
            "pair_index_used": int(pair_index),
            "time_s": float(t_s),
            "flow_mag_median_px_per_frame": mag_median,
            "flow_mag_mean_px_per_frame": mag_mean,
            "flow_mag_p95_px_per_frame": mag_p95,
        })

        prev_gray = gray
        kept += 1

        if MAX_PAIRS is not None and kept >= MAX_PAIRS:
            break

    cap.release()
    df = pd.DataFrame(records)
    return df, float(fps)

def plot_flow_timeseries(df: pd.DataFrame, title: str, out_png: Path):
   
    fig = plt.figure(figsize=(11.2, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(
        df["time_s"],
        df["flow_mag_median_px_per_frame"],
        linewidth=1.8,
        label="Median |Flow| [px/Frame]"
    )

    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Optical Flow |v| [px/Frame]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.title(title)
    export_plot(out_png)

def list_video_files(cam_dir: Path) -> list[Path]:
    vids = []
    for sub in sorted(
        [p for p in cam_dir.iterdir() if p.is_dir()],
        key=lambda p: (parse_frequency_from_folder(p.name) or 10**9, p.name)
    ):
        for vp in sorted(sub.glob("*.mp4")):
            vids.append(vp)
    return vids

if __name__ == "__main__":
    if not CAM2_DIR.exists():
        raise SystemExit(f"CAM2_DIR existiert nicht: {CAM2_DIR}")

    videos = list_video_files(CAM2_DIR)
    if not videos:
        raise SystemExit("Keine MP4s gefunden. Prüfe Ordnerstruktur Cam2\\2Hz\\*.mp4 etc.")

    summary_rows = []

    for vp in videos:
        freq = parse_frequency_from_folder(vp.parent.name)
        signal = parse_signal_from_filename(vp.name) or "Unbekannt"

        tag = f"{vp.parent.name}_{signal}"
        print(f"\n[INFO] {tag} | {vp.name}")

        df_ts, fps = compute_flow_timeseries(vp)
        if df_ts.empty:
            print(f"[WARN] Keine Daten erzeugt: {vp}")
            continue

        out_csv = OUT_DIR / f"{vp.stem}_OF_timeseries.csv"
        df_ts.to_csv(out_csv, index=False)
        print(f"[OK] CSV:  {out_csv}")

        # Plot Flow über Zeit
        title_ts = f"Cam2: Optical Flow über Zeit – {vp.stem}"
        out_png_ts = OUT_DIR / f"{vp.stem}_OF_timeseries.png"
        plot_flow_timeseries(df_ts, title_ts, out_png_ts)

        # Summary Kennwerte 
        flow_med = float(np.median(df_ts["flow_mag_median_px_per_frame"].to_numpy(dtype=float)))
        flow_mean = float(np.mean(df_ts["flow_mag_median_px_per_frame"].to_numpy(dtype=float)))
        flow_p95_med = float(np.median(df_ts["flow_mag_p95_px_per_frame"].to_numpy(dtype=float)))

        summary_rows.append({
            "video": vp.name,
            "folder": vp.parent.name,
            "frequency_hz": int(freq) if freq is not None else np.nan,
            "signal": signal,
            "fps": fps,
            "pairs_used": int(len(df_ts)),
            "flow_median_of_median": flow_med,
            "flow_mean_of_median": flow_mean,
            "flow_median_of_p95": flow_p95_med,
            "roi_frac": ROI_FRAC if USE_CENTER_ROI else np.nan,
            "pair_stride": PAIR_STRIDE
        })

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = OUT_DIR / "Cam2_opticalflow_summary.csv"
    df_sum.to_csv(sum_csv, index=False)
    print(f"\n[OK] Gesamt-Summary: {sum_csv}")
    print(f"[DONE] Alle Cam2 Videos ausgewertet. Ergebnisse in: {OUT_DIR}")
