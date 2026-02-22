from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


CSV_PATH   = Path(r"L:\BA\Messungen\Schlechtweg\camera_front_tele_30fov\camera_front_tele_30fov_sharpness.csv")
VIDEO_PATH = Path(r"L:\BA\Messungen\Schlechtweg\camera_front_tele_30fov.mp4")

OUT_DIR = CSV_PATH.parent / "_Auswertung_Schlechtwegfahrt_Schaerfe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SHARPNESS_COL = "tenengrad"     # oder laplacian_variance
TIME_COL      = "time_s"

SMOOTH_WINDOW_SEC = 0.0        
TARGET_FPS_FOR_PLOT = None

# Zoom-Ausschnitt 
ZOOM_START_S = 0.0
ZOOM_END_S   = 200.0

# Kurven-Ausschnitt 
CURVE_START_S = 3.0
CURVE_END_S   = 12.0

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def export_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Plot: {path}")


def load_sharpness_csv(path: Path) -> pd.DataFrame:
  
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    if TIME_COL not in df.columns:
        raise ValueError(f"{TIME_COL} nicht gefunden. Spalten: {list(df.columns)}")
    if SHARPNESS_COL not in df.columns:
        raise ValueError(f"{SHARPNESS_COL} nicht gefunden. Spalten: {list(df.columns)}")

    df = df[[TIME_COL, SHARPNESS_COL]].rename(
        columns={TIME_COL: "time_s", SHARPNESS_COL: "sharp"}
    )

    df = df.dropna().sort_values("time_s").reset_index(drop=True)
    return df


def estimate_dt(df: pd.DataFrame) -> float:
    t = df["time_s"].to_numpy()
    if len(t) < 3:
        return float("nan")
    return float(np.median(np.diff(t)))


def apply_downsample(df: pd.DataFrame, target_fps: float | None) -> pd.DataFrame:
    if target_fps is None:
        return df

    dt_target = 1.0 / float(target_fps)
    t = df["time_s"].to_numpy()
    sharp = df["sharp"].to_numpy()

    t_new = np.arange(t[0], t[-1], dt_target)
    sharp_new = np.interp(t_new, t, sharp)

    return pd.DataFrame({"time_s": t_new, "sharp": sharp_new})


def apply_smoothing(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    if window_sec <= 0:
        return df

    dt = estimate_dt(df)
    win = int(round(window_sec / dt))
    win = max(1, win)

    s = pd.Series(df["sharp"])
    smooth = s.rolling(window=win, center=True).mean()

    out = df.copy()
    out["sharp"] = smooth
    return out.dropna().reset_index(drop=True)



# Plots
def plot_full_and_zoom(df: pd.DataFrame, title_prefix: str):

    # Gesamtverlauf 
    plt.figure(figsize=(12.5, 4.8))
    plt.plot(df["time_s"], df["sharp"], linewidth=1.0)

    plt.xlabel("Zeit [s]")
    plt.ylabel("Schärfemetrik [a.u.]")
    plt.title(f"{title_prefix} – Gesamtverlauf")
    plt.grid(True, alpha=0.25)

    export_plot(OUT_DIR / "A_schaerfe_full.png")

    # Zoom
    z = df[(df["time_s"] >= ZOOM_START_S) & (df["time_s"] <= ZOOM_END_S)]

    if not z.empty:
        plt.figure(figsize=(12.5, 4.8))
        plt.plot(z["time_s"], z["sharp"], linewidth=1.2)

        plt.xlabel("Zeit [s]")
        plt.ylabel("Schärfemetrik [a.u.]")
        plt.title(f"{title_prefix} – Zoom {ZOOM_START_S:.1f}s–{ZOOM_END_S:.1f}s")
        plt.grid(True, alpha=0.25)

        export_plot(OUT_DIR / "B_schaerfe_zoom.png")


def plot_curve_segment(df: pd.DataFrame, title_prefix: str):

    c = df[(df["time_s"] >= CURVE_START_S) & (df["time_s"] <= CURVE_END_S)]

    if c.empty:
        print("Kurvenfenster leer.")
        return

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(c["time_s"], c["sharp"], linewidth=1.4)

    plt.xlabel("Zeit [s]")
    plt.ylabel("Schärfemetrik [a.u.]")
    plt.title("Schärfeverlauf – Kurvenfahrt")
    plt.grid(True, alpha=0.25)

    export_plot(OUT_DIR / "C_schaerfe_kurve.png")

    # Kennwerte 
    v = c["sharp"].to_numpy()
    print("\nKurven-Ausschnitt Kennwerte:")
    print(f"Median: {np.median(v):.2f}")
    print(f"P05:    {np.quantile(v,0.05):.2f}")
    print(f"P95:    {np.quantile(v,0.95):.2f}")
    print(f"Range:  {np.max(v)-np.min(v):.2f}")

if __name__ == "__main__":

    df = load_sharpness_csv(CSV_PATH)

    df = apply_downsample(df, TARGET_FPS_FOR_PLOT)
    df = apply_smoothing(df, SMOOTH_WINDOW_SEC)

    df.to_csv(OUT_DIR / "sharpness_timeseries_clean.csv", index=False)

    title_prefix = f"Schärfe über Zeit – {CSV_PATH.stem}"

    plot_full_and_zoom(df, title_prefix)
    plot_curve_segment(df, title_prefix)

    print("\n[DONE] Fertig.")
