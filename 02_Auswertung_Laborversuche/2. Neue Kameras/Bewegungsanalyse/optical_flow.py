from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



BASE_CAM2_DIR = Path(r"D:\BA\Messungen\Frequenzen\Videos\Cam2") 
OUT_DIR = BASE_CAM2_DIR / "_Auswertung_OpticalFlow_Cam2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARPNESS_CSV =  Path(r"D:\BA\Messungen\Frequenzen\Videos\normal_Auswertung_Laplace\Cam2\laplacian_summary.csv")  

ROI = None  

MAX_FRAMES = None     
FRAME_STRIDE = 1      

# Optical Flow (Farnebäck)
FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=5,
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

def crop_roi(img, roi):
    if roi is None:
        return img
    x, y, w, h = roi
    H, W = img.shape[:2]
    x0 = max(0, min(W - 1, int(x)))
    y0 = max(0, min(H - 1, int(y)))
    x1 = max(1, min(W, int(x0 + w)))
    y1 = max(1, min(H, int(y0 + h)))
    return img[y0:y1, x0:x1]

def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

def parse_frequency_from_folder(folder_name: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*Hz\s*$", folder_name, flags=re.IGNORECASE)
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

def safe_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        return float("nan")
    return float(fps)


def optical_flow_score(video_path: Path) -> dict:
   
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Kann Video nicht öffnen: {video_path}")

    fps = safe_fps(cap)

    ret, f0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Erstes Frame nicht lesbar: {video_path}")

    g0 = crop_roi(to_gray(f0), ROI)

    mags = []
    frame_count = 0
    used = 0

    while True:
        ret, f = cap.read()
        if not ret:
            break
        frame_count += 1

        if FRAME_STRIDE > 1 and (frame_count % FRAME_STRIDE != 0):
            continue

        g = crop_roi(to_gray(f), ROI)
        flow = cv2.calcOpticalFlowFarneback(g0, g, None, **FB_PARAMS)

        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx * dx + dy * dy)  # px/frame

        mags.append(float(np.median(mag)))  
        used += 1

        g0 = g
        if MAX_FRAMES is not None and used >= MAX_FRAMES:
            break

    cap.release()

    if len(mags) < 5:
        raise RuntimeError(f"Zu wenige ausgewertete Frames: {video_path}")

    mags = np.array(mags, dtype=float)
    score_med = float(np.median(mags))       # main flow score
    score_p90 = float(np.quantile(mags, 0.9))  
    score_rms = float(np.sqrt(np.mean(mags * mags)))

    return {
        "flow_median_mag_px_per_frame": score_med,
        "flow_p90_mag_px_per_frame": score_p90,
        "flow_rms_mag_px_per_frame": score_rms,
        "fps": fps,
        "frames_used": int(len(mags)),
        "frames_total_read": int(frame_count + 1),
    }

rows = []

subdirs = [p for p in BASE_CAM2_DIR.iterdir() if p.is_dir()]
subdirs = sorted(subdirs, key=lambda p: (parse_frequency_from_folder(p.name) or 10**9, p.name))

for d in subdirs:
    if d.name.lower() == "ruhe":
        continue

    freq = parse_frequency_from_folder(d.name)
    if freq is None:
        continue

    mp4s = sorted(d.glob("*.mp4"))
    for vp in mp4s:
        sig = parse_signal_from_filename(vp.name)
        if sig not in ("Sinus", "Rechteck"):
            continue

        print(f"[INFO] Cam2 | {freq} Hz | {sig} | {vp.name}")
        stats = optical_flow_score(vp)

        rows.append({
            "camera": "Cam2",
            "frequency_hz": int(freq),
            "signal": sig,
            "video": vp.name,
            **stats
        })

df_flow = pd.DataFrame(rows)
if df_flow.empty:
    raise SystemExit("Keine Videos gefunden. Prüfe BASE_CAM2_DIR und Dateinamen (Sinus/Rechteck).")

df_flow = df_flow.sort_values(["signal", "frequency_hz"]).reset_index(drop=True)

csv_out = OUT_DIR / "cam2_opticalflow_summary.csv"
df_flow.to_csv(csv_out, index=False)
print(f"[OK] CSV: {csv_out}")

df_sharp = None
sharp_col = None

if SHARPNESS_CSV is not None:
    SHARPNESS_CSV = Path(SHARPNESS_CSV)
    if SHARPNESS_CSV.exists():
        df_sharp = pd.read_csv(SHARPNESS_CSV)

        candidates = [
            "median_laplacian", "mean_laplacian",
            "median_tenengrad", "mean_tenengrad"
        ]
        sharp_col = next((c for c in candidates if c in df_sharp.columns), None)
        if sharp_col is None:
            print(f"[WARN] Schärfe-CSV hat keine bekannte Kennwertspalte. Gefunden: {list(df_sharp.columns)}")
            df_sharp = None
        else:
            df_sharp = df_sharp.copy()
            df_sharp["frequency_hz"] = pd.to_numeric(df_sharp["frequency_hz"], errors="coerce")
            df_sharp = df_sharp.dropna(subset=["frequency_hz"]).copy()
            df_sharp["frequency_hz"] = df_sharp["frequency_hz"].astype(int)

            df_sharp = df_sharp[["frequency_hz", "signal", sharp_col]].copy()

            df_flow = df_flow.merge(df_sharp, on=["frequency_hz", "signal"], how="left")
            print(f"[OK] Schärfe gemerged: {SHARPNESS_CSV.name} -> {sharp_col}")
    else:
        print(f"[WARN] SHARPNESS_CSV nicht gefunden: {SHARPNESS_CSV}")



# normalized flow vs frequency 
FLOW_MAIN = "flow_median_mag_px_per_frame"  

plt.figure(figsize=(8.2, 5.0))

for sig, marker in [("Sinus", "o"), ("Rechteck", "s")]:
    sub = df_flow[df_flow["signal"] == sig].sort_values("frequency_hz").copy()
    if sub.empty:
        continue

    
    ref_flow = float(sub.iloc[0][FLOW_MAIN])
    sub["flow_norm"] = sub[FLOW_MAIN] / ref_flow if ref_flow != 0 else np.nan

    plt.plot(sub["frequency_hz"], sub["flow_norm"], marker=marker, linewidth=2.2,
             label=f"Optical Flow (norm.) – {sig}")

  
    if df_sharp is not None and sharp_col is not None and sharp_col in sub.columns:
        sharp_vals = sub[sharp_col].to_numpy(dtype=float)
        if np.isfinite(sharp_vals).any():
            ref_sh = float(sharp_vals[0])
            sub["sharp_norm"] = sharp_vals / ref_sh if ref_sh != 0 else np.nan
            plt.plot(sub["frequency_hz"], sub["sharp_norm"], marker=marker, linewidth=1.8,
                     linestyle="--", alpha=0.9, label=f"Schärfe (norm.) – {sig}")

plt.axhline(1.0, color="black", linewidth=0.9)
plt.xlabel("Anregungsfrequenz [Hz]")
plt.ylabel("Normiert auf kleinste Frequenz je Signal [-]")
title = "Cam2: Optical Flow vs Frequenz (normiert)"
if df_sharp is not None and sharp_col is not None:
    title += f" + Schärfe-Overlay ({sharp_col})"
plt.title(title)
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, loc="best")

plot_out = OUT_DIR / "Cam2_OF_norm_vs_frequency_singleplot.png"
export_plot(plot_out)


if df_sharp is not None and sharp_col is not None and sharp_col in df_flow.columns:
    valid = df_flow[[FLOW_MAIN, sharp_col]].dropna()
    if len(valid) >= 6:
        corr = np.corrcoef(valid[FLOW_MAIN].to_numpy(), valid[sharp_col].to_numpy())[0, 1]
        print(f"[INFO] Korrelation (alle Punkte gemischt): corr(flow, sharpness) = {corr:.3f}")
        print("       (negativ = mehr Flow -> weniger Schärfe)")
    else:
        print("[WARN] Zu wenige gemergte Punkte für Korrelation.")

print("\n[DONE] Cam2 Optical-Flow-Auswertung fertig.")
print(f"Ergebnisse in: {OUT_DIR}")
