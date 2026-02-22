from __future__ import annotations

import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Baseline (nicht gealtert)
BASELINE_DIR = Path(r"D:\BA\Messungen\Frequenzen\Videos")  
# Gealtert 
ALT_ROOT = Path(r"D:\BA\Messungen\Alt\Frequenz_Videos")             

OUT_DIR = Path(r"D:\BA\Messungen\Alt\Frequenz_Videos\_Auswertung_Laplace_Baseline_vs_Alt_RuheNorm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_FRAC = 0.70
MAX_FRAMES = None
FRAME_STRIDE = 1

USE_MEDIAN_AS_MAIN = True  

EXPORT_EXCEL = True

# Robust gray for sharpness (reduce Bayer/demosaic/chroma influence)
USE_GREEN_CHANNEL = True
GAUSS_BLUR_KSIZE = (3, 3)     # None -> off
GAUSS_BLUR_SIGMA = 0.8        


def to_gray_for_sharpness(frame_bgr: np.ndarray) -> np.ndarray:
    """Green channel + optional light blur to suppress pixel-level artifacts."""
    if USE_GREEN_CHANNEL:
        gray = frame_bgr[:, :, 1].copy()
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if GAUSS_BLUR_KSIZE is not None:
        gray = cv2.GaussianBlur(gray, GAUSS_BLUR_KSIZE, GAUSS_BLUR_SIGMA)

    return gray


def get_center_roi(gray: np.ndarray, frac: float) -> np.ndarray:
    h, w = gray.shape
    frac = float(np.clip(frac, 0.05, 1.0))
    rh = int(h * frac)
    rw = int(w * frac)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    return gray[y0:y0 + rh, x0:x0 + rw]


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def parse_frequency_from_folder(folder_name: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*Hz\s*$", folder_name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_signal_from_name(name: str) -> str | None:
    low = name.lower()
    if "sinus" in low:
        return "Sinus"
    if "rechteck" in low:
        return "Rechteck"
    if "ruhe" in low:
        return "Ruhe"
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

        gray = to_gray_for_sharpness(frame)
        roi = get_center_roi(gray, ROI_FRAC)
        values.append(laplacian_variance(roi))
        kept += 1

        if MAX_FRAMES is not None and kept >= MAX_FRAMES:
            break

    cap.release()
    return values, float(fps), frame_count


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


# Data Collection
def summarize_values(values: list[float]) -> dict:
    v = np.array(values, dtype=float)

    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    iqr = q75 - q25

    return {
        "mean_laplacian": float(v.mean()),
        "std_laplacian": float(v.std(ddof=0)),
        "median_laplacian": float(np.median(v)),
        "q25_laplacian": q25,
        "q75_laplacian": q75,
        "iqr_laplacian": iqr,
        "min_laplacian": float(v.min()),
        "max_laplacian": float(v.max()),
        "frames_used": int(len(values)),
    }


def collect_baseline() -> pd.DataFrame:
    rows = []

    for fdir in sorted([p for p in BASELINE_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        freq = parse_frequency_from_folder(fdir.name)
        is_ruhe_folder = (fdir.name.lower() == "ruhe")

        mp4s = sorted(list(fdir.glob("*.mp4")))
        for vp in mp4s:
            sig = parse_signal_from_name(vp.name)
            
            if is_ruhe_folder:
                sig = "Ruhe"
            if sig not in ("Sinus", "Rechteck", "Ruhe"):
                continue

            values, fps, total_frames = safe_read_video_values(vp)
            if not values:
                continue

            stats = summarize_values(values)
            rows.append({
                "dataset": "Baseline",
                "camera": "Baseline",
                "camera_group": "Baseline",
                "signal": sig,
                "frequency_hz": (0 if sig == "Ruhe" else freq),
                "video": vp.name,
                "roi_frac": ROI_FRAC,
                "fps": fps,
                "frames_total": int(total_frames),
                **stats
            })

    df = pd.DataFrame(rows)
    return df


def collect_alt() -> pd.DataFrame:
    rows = []
 
    for freq_folder in sorted([p for p in ALT_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        freq = parse_frequency_from_folder(freq_folder.name)
        is_ruhe = (freq_folder.name.lower() == "ruhe")

        for cam_dir in sorted([p for p in freq_folder.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            cam_name = cam_dir.name  
            mp4s = sorted(list(cam_dir.glob("*.mp4")))
            for vp in mp4s:
                sig = parse_signal_from_name(vp.name)
                if is_ruhe:
                    sig = "Ruhe"
                if sig not in ("Sinus", "Rechteck", "Ruhe"):
                    continue

                values, fps, total_frames = safe_read_video_values(vp)
                if not values:
                    continue

                stats = summarize_values(values)
                rows.append({
                    "dataset": "Alt",
                    "camera": cam_name,
                    "camera_group": cam_name,
                    "signal": sig,
                    "frequency_hz": (0 if sig == "Ruhe" else freq),
                    "video": vp.name,
                    "roi_frac": ROI_FRAC,
                    "fps": fps,
                    "frames_total": int(total_frames),
                    **stats
                })

    df = pd.DataFrame(rows)
    return df


def add_rest_normalization(df_all: pd.DataFrame, main_col: str) -> pd.DataFrame:
    """
    Adds:
      - rest_value (per dataset+camera based on signal=='Ruhe')
      - rel_to_rest = main_col / rest_value
      - rest_source = 'Ruhe' or 'fallback'
    Fallback for Baseline if no Ruhe video exists: uses the smallest frequency per signal.
    """
    df = df_all.copy()

    rest = df[df["signal"] == "Ruhe"].copy()
    rest_map = (
        rest.groupby(["dataset", "camera_group"], as_index=False)[main_col]
        .median()
        .rename(columns={main_col: "rest_value"})
    )

    df = df.merge(rest_map, on=["dataset", "camera_group"], how="left")
    df["rest_source"] = np.where(df["rest_value"].notna(), "Ruhe", "fallback")

    needs_fallback = (df["dataset"] == "Baseline") & (df["rest_value"].isna()) & (df["signal"].isin(["Sinus", "Rechteck"]))
    if needs_fallback.any():
        for sig in ["Sinus", "Rechteck"]:
            sub = df[(df["dataset"] == "Baseline") & (df["signal"] == sig)].copy()
            if sub.empty:
                continue
            minf = int(sub["frequency_hz"].min())
            ref_val = float(sub.loc[sub["frequency_hz"] == minf, main_col].median())
            df.loc[(df["dataset"] == "Baseline") & (df["signal"] == sig), "rest_value"] = ref_val
            df.loc[(df["dataset"] == "Baseline") & (df["signal"] == sig), "rest_source"] = f"fallback_{minf}Hz"

    df["rel_to_rest"] = df[main_col] / df["rest_value"]
    return df


# Plotting
def plot_absolute_compare(df: pd.DataFrame, signal: str, main_col: str, ylabel: str, out_name: str):
    """
    Absolute curves:
      Baseline + Cam_1..Cam_4 vs frequency for given signal
    """
    sub = df[(df["signal"] == signal) & (df["frequency_hz"] > 0)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(10.5, 5.4))

    for cam in ["Baseline"] + sorted([c for c in sub["camera_group"].unique() if c != "Baseline"]):
        s2 = sub[sub["camera_group"] == cam].sort_values("frequency_hz")
        if s2.empty:
            continue
        plt.plot(s2["frequency_hz"], s2[main_col], marker="o", linewidth=2.0, label=cam)

    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel(ylabel)
    plt.title(f"{signal}: Baseline vs. gealterte Kameras (ROI {int(ROI_FRAC*100)}% zentral)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    export_plot(OUT_DIR / out_name)


def plot_relative_to_rest(df: pd.DataFrame, signal: str, out_name: str):
    """
    Ruhe-normalized curves:
      rel_to_rest vs frequency for each camera incl. baseline
    """
    sub = df[(df["signal"] == signal) & (df["frequency_hz"] > 0)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(10.5, 5.4))

    for cam in ["Baseline"] + sorted([c for c in sub["camera_group"].unique() if c != "Baseline"]):
        s2 = sub[sub["camera_group"] == cam].sort_values("frequency_hz")
        if s2.empty:
            continue
        plt.plot(s2["frequency_hz"], s2["rel_to_rest"], marker="o", linewidth=2.0, label=f"{cam} / Ruhe")

    plt.axhline(1.0, linewidth=1.4)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Relativ zu Ruhe [-]")
    plt.title(f"{signal}: Relativer Schärfeverlust (Wert / Ruhe), ROI {int(ROI_FRAC*100)}%")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    export_plot(OUT_DIR / out_name)


def plot_camera_vs_baseline_ratio(df: pd.DataFrame, signal: str, out_name: str, main_col: str):
    """
    Optional: ratio Cam / Baseline at each frequency (absolute metric ratio).
    This shows how far aged cameras differ from baseline at each frequency.
    """
    sub = df[(df["signal"] == signal) & (df["frequency_hz"] > 0)].copy()
    if sub.empty:
        return

    base = sub[sub["camera_group"] == "Baseline"][["frequency_hz", main_col]].rename(columns={main_col: "baseline_val"})
    if base.empty:
        return

    plt.figure(figsize=(10.5, 5.4))

    for cam in sorted([c for c in sub["camera_group"].unique() if c != "Baseline"]):
        s2 = sub[sub["camera_group"] == cam][["frequency_hz", main_col]].copy()
        s2 = s2.merge(base, on="frequency_hz", how="inner")
        if s2.empty:
            continue
        s2["ratio"] = s2[main_col] / s2["baseline_val"]
        s2 = s2.sort_values("frequency_hz")
        plt.plot(s2["frequency_hz"], s2["ratio"], marker="o", linewidth=2.0, label=f"{cam} / Baseline")

    plt.axhline(1.0, linewidth=1.4)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel("Relativ zur Baseline [-]")
    plt.title(f"{signal}: Relativer Unterschied (Cam / Baseline), ROI {int(ROI_FRAC*100)}%")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    export_plot(OUT_DIR / out_name)


def main():
    main_col = "median_laplacian" if USE_MEDIAN_AS_MAIN else "mean_laplacian"
    main_label = "Median" if USE_MEDIAN_AS_MAIN else "Mittelwert"

    print("[INFO] Sammle Baseline …")
    df_base = collect_baseline()
    if df_base.empty:
        raise SystemExit("Keine Baseline-Videos gefunden. Prüfe BASELINE_DIR und Dateinamen (Sinus/Rechteck).")

    print("[INFO] Sammle Alt (gealtert) …")
    df_alt = collect_alt()
    if df_alt.empty:
        raise SystemExit("Keine Alt-Videos gefunden. Prüfe ALT_ROOT Struktur und Dateinamen.")

    df_all = pd.concat([df_base, df_alt], ignore_index=True)

    df_all = add_rest_normalization(df_all, main_col=main_col)

    df_base.to_csv(OUT_DIR / "laplacian_summary_baseline.csv", index=False)
    df_alt.to_csv(OUT_DIR / "laplacian_summary_alt.csv", index=False)
    df_all.to_csv(OUT_DIR / "laplacian_summary_all_baseline_and_alt_with_rest_norm.csv", index=False)
    print(f"[OK] CSVs gespeichert in: {OUT_DIR}")

    rest_info = (
        df_all[df_all["signal"].isin(["Sinus", "Rechteck"])]
        .groupby(["dataset", "camera_group"], as_index=False)[["rest_value"]]
        .first()
    )
    print("[INFO] Ruhe-Referenzen (rest_value) pro Kamera:")
    print(rest_info.to_string(index=False))

    plot_absolute_compare(
        df=df_all[df_all["signal"].isin(["Sinus", "Rechteck"])],
        signal="Rechteck",
        main_col=main_col,
        ylabel=f"Laplacian-Varianz [-] ({main_label})",
        out_name="V1_compare_absolute_rechteck.png"
    )
    plot_absolute_compare(
        df=df_all[df_all["signal"].isin(["Sinus", "Rechteck"])],
        signal="Sinus",
        main_col=main_col,
        ylabel=f"Laplacian-Varianz [-] ({main_label})",
        out_name="V1_compare_absolute_sinus.png"
    )

    # Relativ zu Ruhe
    plot_relative_to_rest(df_all, "Rechteck", "V3_relative_to_rest_rechteck.png")
    plot_relative_to_rest(df_all, "Sinus", "V3_relative_to_rest_sinus.png")

    # Cam / Baseline
    plot_camera_vs_baseline_ratio(df_all, "Rechteck", "V2_compare_ratio_rechteck.png", main_col=main_col)
    plot_camera_vs_baseline_ratio(df_all, "Sinus", "V2_compare_ratio_sinus.png", main_col=main_col)

    # Absolute (median/mean)
    abs_pivot = (
        df_all[df_all["signal"].isin(["Sinus", "Rechteck"]) & (df_all["frequency_hz"] > 0)]
        .pivot_table(index=["signal", "frequency_hz"], columns=["camera_group"], values=main_col, aggfunc="first")
        .reset_index()
    )
    abs_pivot.to_csv(OUT_DIR / f"laplacian_pivot_absolute_{main_col}.csv", index=False)

    # Relativ zu Ruhe
    rel_pivot = (
        df_all[df_all["signal"].isin(["Sinus", "Rechteck"]) & (df_all["frequency_hz"] > 0)]
        .pivot_table(index=["signal", "frequency_hz"], columns=["camera_group"], values="rel_to_rest", aggfunc="first")
        .reset_index()
    )
    rel_pivot.to_csv(OUT_DIR / "laplacian_pivot_rel_to_rest.csv", index=False)

    if EXPORT_EXCEL:
        try:
            xlsx_path = OUT_DIR / "laplacian_tables_baseline_vs_alt_ruhenorm.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_base.to_excel(writer, sheet_name="baseline_raw", index=False)
                df_alt.to_excel(writer, sheet_name="alt_raw", index=False)
                df_all.to_excel(writer, sheet_name="all_with_rest_norm", index=False)
                abs_pivot.to_excel(writer, sheet_name="pivot_absolute", index=False)
                rel_pivot.to_excel(writer, sheet_name="pivot_rel_to_rest", index=False)
            print(f"[OK] Excel: {xlsx_path}")
        except ModuleNotFoundError:
            print("[WARN] openpyxl nicht installiert -> Excel-Export übersprungen. "
                  "Installieren mit: python -m pip install openpyxl")

    print("\n[DONE] Laplacian-Auswertung Baseline vs Alt inkl. Ruhe-Normalisierung abgeschlossen.")
    print(f"Ergebnisse in: {OUT_DIR}")
    print("\nHinweis:")
    print("- Die Ruhe-Normalisierung ist in rel_to_rest gespeichert (Wert / Ruhe).")
    print("- Für Baseline wird Ruhe genutzt, falls ein Ordner 'Ruhe' existiert; sonst fallback auf kleinste Frequenz pro Signal.")


if __name__ == "__main__":
    main()
