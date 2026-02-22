from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


UM_DIR = Path(r"D:\BA\Messungen\Umgekehrt\Schachbrett") 

OUT_DIR = UM_DIR / "_Auswertung_Laplace_Tenengrad_Umgekehrt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG_LAPLACE_CSV = Path(r"D:\BA\Messungen\Frequenzen\Videos\_Auswertung_Laplace\laplacian_summary.csv")

ORIG_TENENGRAD_CSV = Path(r"D:\BA\Messungen\Frequenzen\Videos\_Auswertung_Tenengrad\tenengrad_summary.csv")


ROI = (916, 238, 2054, 1416)  


MAX_FRAMES = None         
FRAME_STRIDE = 1           

SAVE_PER_FRAME_CSV = False
MAKE_DISTRIBUTION_PLOTS = True
MAKE_TIMESERIES = True
USE_MEDIAN_AS_MAIN = True
EXPORT_EXCEL = True

# Bayer/Demosaic Stabilisierung
USE_GREEN_CHANNEL = True
GAUSS_BLUR_KSIZE = (3, 3)     # None zum Abschalten
GAUSS_BLUR_SIGMA = 0.8

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
    print(f"[OK] Plot: {path}")
    plt.close()


def to_gray_for_sharpness(frame_bgr: np.ndarray) -> np.ndarray:
    if USE_GREEN_CHANNEL:
        gray = frame_bgr[:, :, 1].copy()
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if GAUSS_BLUR_KSIZE is not None:
        gray = cv2.GaussianBlur(gray, GAUSS_BLUR_KSIZE, GAUSS_BLUR_SIGMA)

    return gray


def crop_roi(gray: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = roi
    H, W = gray.shape
    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    x1 = int(np.clip(x0 + w, 1, W))
    y1 = int(np.clip(y0 + h, 1, H))
    return gray[y0:y1, x0:x1]


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    return float(np.mean(mag2))


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


def safe_read_video_values(video_path: Path) -> tuple[list[float], list[float], float, int]:
   
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video kann nicht geöffnet werden: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float("nan")

    lap_vals: list[float] = []
    ten_vals: list[float] = []
    frame_count = 0
    kept = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if FRAME_STRIDE > 1 and (frame_count - 1) % FRAME_STRIDE != 0:
            continue

        gray = to_gray_for_sharpness(frame)
        roi_img = crop_roi(gray, ROI)

        lap_vals.append(laplacian_variance(roi_img))
        ten_vals.append(tenengrad(roi_img))
        kept += 1

        if MAX_FRAMES is not None and kept >= MAX_FRAMES:
            break

    cap.release()
    return lap_vals, ten_vals, float(fps), int(frame_count)


def summarize_distribution(v: np.ndarray) -> dict:
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=0)),
        "median": float(np.median(v)),
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def pick_main_col(df_in: pd.DataFrame, prefix: str, prefer_median: bool) -> str:

    med = f"median_{prefix}"
    mean = f"mean_{prefix}"
    if prefer_median and med in df_in.columns:
        return med
    if mean in df_in.columns:
        return mean
    for c in df_in.columns:
        if c.endswith(f"_{prefix}") and ("median" in c or "mean" in c):
            return c
    raise KeyError(f"Kein Hauptkennwert für prefix='{prefix}' gefunden.")


def ensure_freq_num(df_in: pd.DataFrame) -> pd.DataFrame:
 
    df = df_in.copy()
    if "frequency_hz_num" not in df.columns:
        if "frequency_hz" in df.columns:
            df["frequency_hz_num"] = pd.to_numeric(df["frequency_hz"], errors="coerce")
        else:
            raise KeyError("Weder 'frequency_hz_num' noch 'frequency_hz' in CSV gefunden.")
    else:
        df["frequency_hz_num"] = pd.to_numeric(df["frequency_hz_num"], errors="coerce")

    df = df[~df["frequency_hz_num"].isna()].copy()
    return df


def norm_by_smallest_freq_per_signal(df_in: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = []
    for sig in ["Sinus", "Rechteck"]:
        sub = df_in[df_in["signal"] == sig].sort_values("frequency_hz_num").copy()
        if sub.empty:
            continue
        ref = float(sub.iloc[0][value_col])
        sub["norm"] = sub[value_col] / ref if ref != 0 else np.nan
        out.append(sub)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

rows_summary = []
per_frame_records = []

subdirs = [p for p in UM_DIR.iterdir() if p.is_dir()]

def _sort_key(p: Path):
    if p.name.lower() == "ruhe":
        return (-1, p.name)
    f = parse_frequency_from_folder(p.name)
    return (f if f is not None else 10**9, p.name)

subdirs = sorted(subdirs, key=_sort_key)

for d in subdirs:
    if d.name.lower() == "ruhe":
        mp4s = sorted(list(d.glob("*.mp4")))
        for vp in mp4s:
            print(f"[INFO] Ruhe | {vp.name}")
            lap_vals, ten_vals, fps, total_frames = safe_read_video_values(vp)
            if len(lap_vals) == 0:
                print(f"[WARN] Keine Frames ausgewertet: {vp}")
                continue

            lap = np.array(lap_vals, dtype=float)
            ten = np.array(ten_vals, dtype=float)
            s_lap = summarize_distribution(lap)
            s_ten = summarize_distribution(ten)

            rows_summary.append({
                "measurement": "Umgekehrt",
                "frequency_hz": "",
                "frequency_hz_num": np.nan,
                "signal": "Ruhe",
                "video": vp.name,
                "fps": fps,
                "frames_total": int(total_frames),
                "frames_used": int(len(lap_vals)),
                "roi_x": ROI[0], "roi_y": ROI[1], "roi_w": ROI[2], "roi_h": ROI[3],

                "mean_laplacian": s_lap["mean"],
                "std_laplacian": s_lap["std"],
                "median_laplacian": s_lap["median"],
                "q25_laplacian": s_lap["q25"],
                "q75_laplacian": s_lap["q75"],
                "iqr_laplacian": s_lap["iqr"],
                "min_laplacian": s_lap["min"],
                "max_laplacian": s_lap["max"],

                "mean_tenengrad": s_ten["mean"],
                "std_tenengrad": s_ten["std"],
                "median_tenengrad": s_ten["median"],
                "q25_tenengrad": s_ten["q25"],
                "q75_tenengrad": s_ten["q75"],
                "iqr_tenengrad": s_ten["iqr"],
                "min_tenengrad": s_ten["min"],
                "max_tenengrad": s_ten["max"],
            })

            if SAVE_PER_FRAME_CSV:
                for i, (lv, tv) in enumerate(zip(lap_vals, ten_vals)):
                    per_frame_records.append({
                        "measurement": "Umgekehrt",
                        "frequency_hz": "",
                        "signal": "Ruhe",
                        "video": vp.name,
                        "frame_index_used": i,
                        "laplacian": float(lv),
                        "tenengrad": float(tv),
                    })
        continue

    freq = parse_frequency_from_folder(d.name)
    if freq is None:
        continue

    mp4s = sorted(list(d.glob("*.mp4")))
    for vp in mp4s:
        signal = parse_signal_from_filename(vp.name)
        if signal not in ("Sinus", "Rechteck"):
            continue

        print(f"[INFO] {freq} Hz | {signal} | {vp.name}")
        lap_vals, ten_vals, fps, total_frames = safe_read_video_values(vp)
        if len(lap_vals) == 0:
            print(f"[WARN] Keine Frames ausgewertet: {vp}")
            continue

        lap = np.array(lap_vals, dtype=float)
        ten = np.array(ten_vals, dtype=float)
        s_lap = summarize_distribution(lap)
        s_ten = summarize_distribution(ten)

        rows_summary.append({
            "measurement": "Umgekehrt",
            "frequency_hz": int(freq),
            "frequency_hz_num": float(freq),
            "signal": signal,
            "video": vp.name,
            "fps": fps,
            "frames_total": int(total_frames),
            "frames_used": int(len(lap_vals)),
            "roi_x": ROI[0], "roi_y": ROI[1], "roi_w": ROI[2], "roi_h": ROI[3],

            "mean_laplacian": s_lap["mean"],
            "std_laplacian": s_lap["std"],
            "median_laplacian": s_lap["median"],
            "q25_laplacian": s_lap["q25"],
            "q75_laplacian": s_lap["q75"],
            "iqr_laplacian": s_lap["iqr"],
            "min_laplacian": s_lap["min"],
            "max_laplacian": s_lap["max"],

            "mean_tenengrad": s_ten["mean"],
            "std_tenengrad": s_ten["std"],
            "median_tenengrad": s_ten["median"],
            "q25_tenengrad": s_ten["q25"],
            "q75_tenengrad": s_ten["q75"],
            "iqr_tenengrad": s_ten["iqr"],
            "min_tenengrad": s_ten["min"],
            "max_tenengrad": s_ten["max"],
        })

        if SAVE_PER_FRAME_CSV:
            for i, (lv, tv) in enumerate(zip(lap_vals, ten_vals)):
                per_frame_records.append({
                    "measurement": "Umgekehrt",
                    "frequency_hz": int(freq),
                    "signal": signal,
                    "video": vp.name,
                    "frame_index_used": i,
                    "laplacian": float(lv),
                    "tenengrad": float(tv),
                })

df = pd.DataFrame(rows_summary)
if df.empty:
    raise SystemExit("Keine Videos gefunden/ausgewertet. Prüfe UM_DIR und Dateinamen (Um_Sinus_2Hz etc.).")

df_freq = df.copy()
df_freq["frequency_hz_num"] = pd.to_numeric(df_freq["frequency_hz"], errors="coerce")
df_freq = df_freq[~df_freq["frequency_hz_num"].isna()].copy()
df_freq = df_freq.sort_values(["signal", "frequency_hz_num"]).reset_index(drop=True)

summary_csv = OUT_DIR / "laplace_tenengrad_summary_umgekehrt.csv"
df.to_csv(summary_csv, index=False)
print(f"[OK] Summary CSV: {summary_csv}")

if SAVE_PER_FRAME_CSV and per_frame_records:
    df_pf = pd.DataFrame(per_frame_records)
    pf_csv = OUT_DIR / "laplace_tenengrad_per_frame_umgekehrt.csv"
    df_pf.to_csv(pf_csv, index=False)
    print(f"[OK] Per-frame CSV: {pf_csv}")

def plot_standard_set(metric_prefix: str, metric_label: str, filename_prefix: str):
    mean_col = f"mean_{metric_prefix}"
    std_col  = f"std_{metric_prefix}"
    med_col  = f"median_{metric_prefix}"

    MAIN_COL   = med_col if USE_MEDIAN_AS_MAIN else mean_col
    MAIN_LABEL = "Median" if USE_MEDIAN_AS_MAIN else "Mittelwert"

    # Combined: Sinus vs Rechteck + error bars
    plt.figure(figsize=(7.6, 4.6))
    for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
        sub = df_freq[df_freq["signal"] == signal].sort_values("frequency_hz_num")
        if sub.empty:
            continue
        plt.errorbar(
            sub["frequency_hz_num"],
            sub[MAIN_COL],
            yerr=sub[std_col],
            fmt=marker + "-",
            capsize=4,
            label=signal
        )
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel(f"{metric_label} ({MAIN_LABEL})")
    plt.title(f"Umgekehrt: {metric_label} vs. Frequenz (ROI fix)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    export_plot(OUT_DIR / f"A_{filename_prefix}_vs_frequency_errorbars.png")

    # Normalized per signal 
    plt.figure(figsize=(7.6, 4.6))
    for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
        sub = df_freq[df_freq["signal"] == signal].sort_values("frequency_hz_num").copy()
        if sub.empty:
            continue
        ref = float(sub.iloc[0][MAIN_COL])
        sub["norm"] = sub[MAIN_COL] / ref if ref != 0 else np.nan
        plt.plot(sub["frequency_hz_num"], sub["norm"], marker + "-", label=signal)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Normiert [-] (zu kleinster Frequenz)")
    plt.title("Relativer Verlauf (je Signal normiert auf kleinste Frequenz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    export_plot(OUT_DIR / f"B_{filename_prefix}_normalized.png")

    # Mean - Median 
    plt.figure(figsize=(7.6, 4.6))
    for signal, marker in zip(["Sinus", "Rechteck"], ["o", "s"]):
        sub = df_freq[df_freq["signal"] == signal].sort_values("frequency_hz_num")
        if sub.empty:
            continue
        delta = sub[mean_col] - sub[med_col]
        plt.plot(sub["frequency_hz_num"], delta, marker + "-", label=signal)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Mean − Median")
    plt.title("Diagnose: Ausreißer-/Rausch-Einfluss (Mean vs. Median)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    export_plot(OUT_DIR / f"C_{filename_prefix}_mean_minus_median.png")

    # Separate curves per signal
    for signal in ["Sinus", "Rechteck"]:
        sub = df_freq[df_freq["signal"] == signal].sort_values("frequency_hz_num")
        if sub.empty:
            continue
        plt.figure(figsize=(7.6, 4.6))
        plt.errorbar(
            sub["frequency_hz_num"],
            sub[MAIN_COL],
            yerr=sub[std_col],
            fmt="o-",
            capsize=4
        )
        plt.xlabel("Anregungsfrequenz [Hz]")
        plt.ylabel(f"{metric_label} ({MAIN_LABEL})")
        plt.title(f"{signal}: {metric_label} vs. Frequenz (ROI fix)")
        plt.grid(True, alpha=0.3)
        export_plot(OUT_DIR / f"D_{signal.lower()}_{filename_prefix}_vs_frequency.png")

    # Violin distribution
    def make_violin_distribution(signal: str):
        freqs = sorted(df_freq["frequency_hz_num"].unique())
        distributions = []
        medians = []
        q25s = []
        q75s = []
        means = []
        used_freqs = []

        for freq in freqs:
            folder = UM_DIR / f"{int(freq)}Hz"
            if not folder.exists():
                continue

            cand = list(folder.glob(f"*{signal}*{int(freq)}Hz*.mp4"))
            if not cand:
                cand = list(folder.glob("*.mp4"))
            if not cand:
                continue

            vp = sorted(cand)[0]
            lap_vals, ten_vals, _, _ = safe_read_video_values(vp)
            arr = np.array(lap_vals if metric_prefix == "laplacian" else ten_vals, dtype=float)
            if arr.size == 0:
                continue

            distributions.append(arr)
            medians.append(float(np.median(arr)))
            q25s.append(float(np.quantile(arr, 0.25)))
            q75s.append(float(np.quantile(arr, 0.75)))
            means.append(float(np.mean(arr)))
            used_freqs.append(int(freq))

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
        plt.ylabel(f"{metric_label} (Frame-weise Verteilung)")
        plt.title(f"{signal}: Verteilung je Frequenz (Violin + Median + IQR), ROI fix")
        plt.grid(True, alpha=0.25, axis="y")
        plt.legend(ncol=3, loc="upper left")
        export_plot(OUT_DIR / f"E_{signal.lower()}_{filename_prefix}_violin_distribution.png")

    if MAKE_DISTRIBUTION_PLOTS:
        make_violin_distribution("Sinus")
        make_violin_distribution("Rechteck")

    # Time series per video 
    if MAKE_TIMESERIES:
        for _, row in df_freq.iterrows():
            freq = int(row["frequency_hz_num"])
            signal = str(row["signal"])
            folder = UM_DIR / f"{freq}Hz"
            vp = folder / str(row["video"])
            if not vp.exists():
                continue

            lap_vals, ten_vals, fps, _ = safe_read_video_values(vp)
            vals = lap_vals if metric_prefix == "laplacian" else ten_vals
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
            plt.ylabel(metric_label)
            plt.title(f"Zeitverlauf: {signal} {freq} Hz (ROI fix)")
            plt.grid(True, alpha=0.25)
            export_plot(OUT_DIR / f"F_timeseries_{filename_prefix}_{signal.lower()}_{freq}Hz.png")


# Standard-Plots (Umgekehrt)
plot_standard_set("laplacian", "Laplacian-Varianz [-]", "laplace")
plot_standard_set("tenengrad", "Tenengrad (Gradientenenergie) [-]", "tenengrad")


MAIN_LAP = "median_laplacian" if USE_MEDIAN_AS_MAIN else "mean_laplacian"
MAIN_TEN = "median_tenengrad" if USE_MEDIAN_AS_MAIN else "mean_tenengrad"

pivot_lap = df_freq.pivot_table(index="frequency_hz_num", columns="signal", values=MAIN_LAP, aggfunc="first").reset_index()
pivot_ten = df_freq.pivot_table(index="frequency_hz_num", columns="signal", values=MAIN_TEN, aggfunc="first").reset_index()

pivot_lap.to_csv(OUT_DIR / f"pivot_umgekehrt_{MAIN_LAP}.csv", index=False)
pivot_ten.to_csv(OUT_DIR / f"pivot_umgekehrt_{MAIN_TEN}.csv", index=False)
print(f"[OK] Pivot CSV Laplace: {OUT_DIR / f'pivot_umgekehrt_{MAIN_LAP}.csv'}")
print(f"[OK] Pivot CSV Tenengrad: {OUT_DIR / f'pivot_umgekehrt_{MAIN_TEN}.csv'}")

if EXPORT_EXCEL:
    try:
        xlsx_path = OUT_DIR / "tables_umgekehrt_laplace_tenengrad.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="summary_all", index=False)
            df_freq.to_excel(writer, sheet_name="summary_freq_only", index=False)
            pivot_lap.to_excel(writer, sheet_name="pivot_laplace_main", index=False)
            pivot_ten.to_excel(writer, sheet_name="pivot_tenengrad_main", index=False)
        print(f"[OK] Excel: {xlsx_path}")
    except ModuleNotFoundError:
        print("[WARN] openpyxl nicht installiert -> Excel-Export übersprungen. Installieren: python -m pip install openpyxl")


def make_metric_comparison(metric_prefix: str, metric_title: str, orig_csv: Path, out_prefix: str):
    if not orig_csv.exists():
        print(f"[WARN] Original CSV nicht gefunden ({metric_prefix}): {orig_csv}")
        print(f"[WARN] Vergleichsplots für {metric_prefix} werden übersprungen.")
        return

    df_orig = pd.read_csv(orig_csv)


    df_orig = ensure_freq_num(df_orig)

    if "signal" not in df_orig.columns:
        raise KeyError(f"Original CSV ({orig_csv}) hat keine Spalte 'signal'.")

    df_o = df_orig.copy()
    df_u = df_freq.copy()

    main_o = pick_main_col(df_o, metric_prefix, USE_MEDIAN_AS_MAIN)
    main_u = pick_main_col(df_u, metric_prefix, USE_MEDIAN_AS_MAIN)

    # Absolute Vergleich
    plt.figure(figsize=(8.2, 5.0))
    for signal, marker in [("Sinus", "o"), ("Rechteck", "s")]:
        so = df_o[df_o["signal"] == signal].sort_values("frequency_hz_num")
        su = df_u[df_u["signal"] == signal].sort_values("frequency_hz_num")
        if not so.empty:
            plt.plot(so["frequency_hz_num"], so[main_o], marker=marker, linewidth=2.0,
                     label=f"{signal} – Kamera bewegt")
        if not su.empty:
            plt.plot(su["frequency_hz_num"], su[main_u], marker=marker, linewidth=2.0, linestyle="--",
                     label=f"{signal} – Target bewegt")
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel(f"{metric_title} (Hauptkennwert)")
    plt.title(f"Vergleich (absolut): Kamera bewegt vs. Target bewegt – {metric_title}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, loc="best")
    export_plot(OUT_DIR / f"COMP_abs_{out_prefix}_kamera_vs_target.png")

    # Relativ: je Signal auf kleinste Frequenz normiert 
    no = norm_by_smallest_freq_per_signal(df_o, main_o)
    nu = norm_by_smallest_freq_per_signal(df_u, main_u)

    plt.figure(figsize=(8.2, 5.0))
    for sig in ["Sinus", "Rechteck"]:
        so = no[no["signal"] == sig].sort_values("frequency_hz_num")
        su = nu[nu["signal"] == sig].sort_values("frequency_hz_num")
        if not so.empty:
            plt.plot(so["frequency_hz_num"], so["norm"], "-o", linewidth=2.0, label=f"{sig} – Kamera bewegt")
        if not su.empty:
            plt.plot(su["frequency_hz_num"], su["norm"], "--o", linewidth=2.0, label=f"{sig} – Target bewegt")

    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Relativ zu Referenz [-]")
    plt.title(f"Vergleich (relativ): Kamera bewegt vs. Target bewegt – {metric_title} (normiert)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, loc="best")
    export_plot(OUT_DIR / f"COMP_rel_{out_prefix}_kamera_vs_target.png")


# Laplace-Vergleich
make_metric_comparison(
    metric_prefix="laplacian",
    metric_title="Laplacian-Varianz [-]",
    orig_csv=ORIG_LAPLACE_CSV,
    out_prefix="laplace"
)

# Tenengrad-Vergleich 
make_metric_comparison(
    metric_prefix="tenengrad",
    metric_title="Tenengrad (Gradientenenergie) [-]",
    orig_csv=ORIG_TENENGRAD_CSV,
    out_prefix="tenengrad"
)

print("\n[DONE] Umgekehrt-Auswertung + Vergleich (Laplace & Tenengrad) abgeschlossen.")
print(f"Ergebnisse in: {OUT_DIR}")
