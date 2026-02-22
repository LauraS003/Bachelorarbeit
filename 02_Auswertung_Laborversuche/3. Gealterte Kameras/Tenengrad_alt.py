from __future__ import annotations

import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"D:\BA\Messungen\Frequenzen\Videos")  
OUT_DIR  = Path(r"D:\BA\Messungen\Frequenzen\Videos\_Auswertung_Tenengrad")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_FRAC = 0.70            
MAX_FRAMES = None         
FRAME_STRIDE = 1           

SAVE_PER_FRAME_CSV = False 
MAKE_DISTRIBUTION_PLOTS = True  
MAKE_TIMESERIES = True          
USE_MEDIAN_AS_MAIN = True       # Hauptkennwert 

EXPORT_EXCEL = True        

USE_GREEN_CHANNEL = True    
APPLY_PREBLUR = True         
PREBLUR_KSIZE = 3           
PREBLUR_SIGMA = 0.8         


def get_center_roi(gray: np.ndarray, frac: float) -> np.ndarray:
   
    h, w = gray.shape
    frac = float(np.clip(frac, 0.05, 1.0))
    rh = int(h * frac)
    rw = int(w * frac)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    return gray[y0:y0 + rh, x0:x0 + rw]

def frame_to_gray(frame_bgr: np.ndarray) -> np.ndarray:

    if USE_GREEN_CHANNEL:
        gray = frame_bgr[:, :, 1].copy()
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if APPLY_PREBLUR:
        k = int(PREBLUR_KSIZE)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), PREBLUR_SIGMA)

    return gray

def tenengrad(gray: np.ndarray) -> float:
 
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))

def parse_frequency_from_folder(folder_name: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*Hz\s*$", folder_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_signal_from_filename(fname: str) -> str | None:
    low = fname.lower()
    if "sinus" in low:
        return "Sinus"
    if "rechteck" in low:
        return "Rechteck"
    return None

def safe_read_video_values(video_path: Path) -> tuple[list[float], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video kann nicht geöffnet werden: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float("nan")

    values: list[float] = []
    frame_count = 0
    kept = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if FRAME_STRIDE > 1 and (frame_count - 1) % FRAME_STRIDE != 0:
            continue

        gray = frame_to_gray(frame)
        roi = get_center_roi(gray, ROI_FRAC)
        values.append(tenengrad(roi))
        kept += 1

        if MAX_FRAMES is not None and kept >= MAX_FRAMES:
            break

    cap.release()
    return values, float(fps), frame_count


def export_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"[OK] Plot: {path}")
    plt.close()

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

rows_summary = []
per_frame_records = []

freq_dirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]
freq_dirs = sorted(freq_dirs, key=lambda p: (parse_frequency_from_folder(p.name) or 10**9, p.name))

for fdir in freq_dirs:
    freq = parse_frequency_from_folder(fdir.name)
    if freq is None:
        continue

    mp4s = sorted(list(fdir.glob("*.mp4")))
    for vp in mp4s:
        signal = parse_signal_from_filename(vp.name)
        if signal is None:
            continue

        print(f"[INFO] {freq} Hz | {signal} | {vp.name}")

        values, fps, total_frames = safe_read_video_values(vp)
        if len(values) == 0:
            print(f"[WARN] Keine Frames ausgewertet: {vp}")
            continue

        v = np.array(values, dtype=float)

        q25 = float(np.quantile(v, 0.25))
        q75 = float(np.quantile(v, 0.75))
        iqr = q75 - q25

        rows_summary.append({
            "frequency_hz": freq,
            "signal": signal,
            "video": vp.name,
            "roi_frac": ROI_FRAC,
            "fps": fps,
            "frames_total": int(total_frames),
            "frames_used": int(len(values)),
            "mean_tenengrad": float(v.mean()),
            "std_tenengrad": float(v.std(ddof=0)),
            "median_tenengrad": float(np.median(v)),
            "q25_tenengrad": q25,
            "q75_tenengrad": q75,
            "iqr_tenengrad": iqr,
            "min_tenengrad": float(v.min()),
            "max_tenengrad": float(v.max()),
            "use_green_channel": bool(USE_GREEN_CHANNEL),
            "preblur": bool(APPLY_PREBLUR),
            "preblur_ksize": int(PREBLUR_KSIZE) if APPLY_PREBLUR else 0,
            "preblur_sigma": float(PREBLUR_SIGMA) if APPLY_PREBLUR else 0.0,
        })

        if SAVE_PER_FRAME_CSV:
            for i, val in enumerate(values):
                per_frame_records.append({
                    "frequency_hz": freq,
                    "signal": signal,
                    "video": vp.name,
                    "frame_index_used": i,
                    "tenengrad": float(val),
                })

df = pd.DataFrame(rows_summary)
if df.empty:
    raise SystemExit("Keine Videos gefunden/ausgewertet. Prüfe BASE_DIR und Dateinamen (Sinus_/Rechteck_).")

df = df.sort_values(["signal", "frequency_hz"]).reset_index(drop=True)

summary_csv = OUT_DIR / "tenengrad_summary.csv"
df.to_csv(summary_csv, index=False)
print(f"[OK] Summary CSV: {summary_csv}")

if SAVE_PER_FRAME_CSV and per_frame_records:
    df_pf = pd.DataFrame(per_frame_records)
    pf_csv = OUT_DIR / "tenengrad_per_frame_all.csv"
    df_pf.to_csv(pf_csv, index=False)
    print(f"[OK] Per-frame CSV: {pf_csv}")

MAIN_COL = "median_tenengrad" if USE_MEDIAN_AS_MAIN else "mean_tenengrad"
MAIN_LABEL = "Median" if USE_MEDIAN_AS_MAIN else "Mittelwert"

# Combined: Sinus vs Rechteck + error bars
plt.figure(figsize=(7.6, 4.6))
for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
    sub = df[df["signal"] == signal].sort_values("frequency_hz")
    if sub.empty:
        continue
    plt.errorbar(
        sub["frequency_hz"],
        sub[MAIN_COL],
        yerr=sub["std_tenengrad"],
        fmt=marker + "-",
        capsize=4,
        label=signal
    )
plt.xlabel("Anregungsfrequenz [Hz]")
plt.ylabel(f"Tenengrad [-] ({MAIN_LABEL})")
plt.title(f"Schärfemaß (Tenengrad) vs. Frequenz (ROI {int(ROI_FRAC*100)}% zentral)")
plt.grid(True, alpha=0.3)
plt.legend()
export_plot(OUT_DIR / "A_tenengrad_vs_frequency_errorbars.png")

# Normalized per signal
plt.figure(figsize=(7.6, 4.6))
for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
    sub = df[df["signal"] == signal].sort_values("frequency_hz").copy()
    if sub.empty:
        continue
    ref = float(sub.iloc[0][MAIN_COL])
    sub["norm"] = sub[MAIN_COL] / ref if ref != 0 else np.nan
    plt.plot(sub["frequency_hz"], sub["norm"], marker + "-", label=signal)
plt.xlabel("Anregungsfrequenz [Hz]")
plt.ylabel(f"Normierter Tenengrad [-] ({MAIN_LABEL}/Referenz)")
plt.title("Relativer Schärfeverlauf (je Signal normiert auf kleinste Frequenz)")
plt.grid(True, alpha=0.3)
plt.legend()
export_plot(OUT_DIR / "B_tenengrad_normalized.png")

# Mean - Median 
plt.figure(figsize=(7.6, 4.6))
for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
    sub = df[df["signal"] == signal].sort_values("frequency_hz")
    if sub.empty:
        continue
    delta = sub["mean_tenengrad"] - sub["median_tenengrad"]
    plt.plot(sub["frequency_hz"], delta, marker + "-", label=signal)
plt.axhline(0, linewidth=0.8)
plt.xlabel("Anregungsfrequenz [Hz]")
plt.ylabel("Mean − Median")
plt.title("Diagnose: Ausreißer-/Rausch-Einfluss (Mean vs. Median)")
plt.grid(True, alpha=0.3)
plt.legend()
export_plot(OUT_DIR / "C_tenengrad_mean_minus_median.png")

# Separate curves per signal
for signal in ["Sinus", "Rechteck"]:
    sub = df[df["signal"] == signal].sort_values("frequency_hz")
    if sub.empty:
        continue
    plt.figure(figsize=(7.6, 4.6))
    plt.errorbar(
        sub["frequency_hz"],
        sub[MAIN_COL],
        yerr=sub["std_tenengrad"],
        fmt="o-",
        capsize=4
    )
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel(f"Tenengrad [-] ({MAIN_LABEL})")
    plt.title(f"{signal}: Schärfemaß vs. Frequenz (ROI {int(ROI_FRAC*100)}% zentral)")
    plt.grid(True, alpha=0.3)
    export_plot(OUT_DIR / f"D_{signal.lower()}_tenengrad_vs_frequency.png")


# Violin + Median + IQR
def make_violin_distribution(signal: str):
    freqs = sorted(df["frequency_hz"].unique())

    distributions = []
    medians = []
    q25s = []
    q75s = []
    means = []
    used_freqs = []

    for freq in freqs:
        folder = BASE_DIR / f"{freq}Hz"
        vp = folder / f"{signal}_{freq}Hz.mp4"
        if not vp.exists():
            cand = list(folder.glob(f"*{signal}*{freq}Hz*.mp4"))
            if not cand:
                continue
            vp = cand[0]

        vals, _, _ = safe_read_video_values(vp)
        if len(vals) == 0:
            continue

        v = np.array(vals, dtype=float)
        distributions.append(v)
        medians.append(float(np.median(v)))
        q25s.append(float(np.quantile(v, 0.25)))
        q75s.append(float(np.quantile(v, 0.75)))
        means.append(float(v.mean()))
        used_freqs.append(freq)

    if len(distributions) < 2:
        return

    positions = np.arange(1, len(distributions) + 1)

    plt.figure(figsize=(10.5, 4.8))

    parts = plt.violinplot(
        distributions,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for pc in parts["bodies"]:
        pc.set_alpha(0.35)

    plt.vlines(positions, q25s, q75s, linewidth=3, alpha=0.9, label="IQR (Q25–Q75)")
    plt.scatter(positions, medians, marker="o", s=35, label="Median")
    plt.scatter(positions, means, marker="x", s=35, label="Mean", alpha=0.9)

    plt.xticks(positions, [str(f) for f in used_freqs])
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Tenengrad [-] (Frame-weise Verteilung)")
    plt.title(f"{signal}: Verteilung je Frequenz (Violin + Median + IQR), ROI {int(ROI_FRAC*100)}%")
    plt.grid(True, alpha=0.25, axis="y")
    plt.legend(ncol=3, loc="upper left")
    export_plot(OUT_DIR / f"E_{signal.lower()}_tenengrad_violin_distribution.png")

if MAKE_DISTRIBUTION_PLOTS:
    make_violin_distribution("Sinus")
    make_violin_distribution("Rechteck")


# Time series per video 
if MAKE_TIMESERIES:
    for _, row in df.iterrows():
        freq = int(row["frequency_hz"])
        signal = str(row["signal"])
        folder = BASE_DIR / f"{freq}Hz"
        vp = folder / row["video"]
        if not vp.exists():
            continue

        vals, fps, _ = safe_read_video_values(vp)
        if len(vals) < 10:
            continue

        if not np.isnan(fps) and fps > 0:
            t = np.arange(len(vals)) / fps * FRAME_STRIDE
            xlab = "Zeit [s]"
        else:
            t = np.arange(len(vals))
            xlab = "Frame-Index"

        plt.figure(figsize=(10.5, 3.8))
        plt.plot(t, vals, linewidth=0.9)
        plt.xlabel(xlab)
        plt.ylabel("Tenengrad [-]")
        plt.title(f"Zeitverlauf: {signal} {freq} Hz (ROI {int(ROI_FRAC*100)}% zentral)")
        plt.grid(True, alpha=0.25)
        export_plot(OUT_DIR / f"F_timeseries_{signal.lower()}_{freq}Hz_tenengrad.png")


pivot = df.pivot_table(index="frequency_hz", columns="signal", values=MAIN_COL, aggfunc="first").reset_index()
pivot_csv = OUT_DIR / f"tenengrad_pivot_{MAIN_COL}.csv"
pivot.to_csv(pivot_csv, index=False)
print(f"[OK] Pivot CSV: {pivot_csv}")

if EXPORT_EXCEL:
    try:
        xlsx_path = OUT_DIR / "tenengrad_tables.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="summary", index=False)
            pivot.to_excel(writer, sheet_name="pivot_main_metric", index=False)
        print(f"[OK] Excel: {xlsx_path}")
    except ModuleNotFoundError:
        print("[WARN] openpyxl nicht installiert -> Excel-Export übersprungen. "
              "Installieren mit: python -m pip install openpyxl")

print("\n[DONE] Tenengrad-Auswertung abgeschlossen.")
print(f"Ergebnisse in: {OUT_DIR}")
