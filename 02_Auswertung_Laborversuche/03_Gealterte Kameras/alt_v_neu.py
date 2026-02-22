from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


CSV_ALT = Path(r"L:\BA\Messungen\Alt\Frequenz_Videos\alt_Auswertung_Laplace\laplacian_summary_alt.csv")
CSV_NEU = Path(r"L:\BA\Messungen\Frequenzen\Videos\normal_Auswertung_Laplace\_Combined\laplacian_summary_all_cams.csv")

OUT_DIR = Path(r"L:\BA\Messungen\_Vergleich_ALT_vs_NEU_Tabelle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_COL = "median_laplacian"     # ggf. auf mean_laplacian ändern

# Relative Normierung:
# Wenn REF_FREQ_HZ gesetzt und vorhanden, darauf normieren
# sonst automatisch kleinste Frequenz pro (Kamera, Signal)
REF_FREQ_HZ: int | None = 2

# Range:
ROBUST_LOW_Q  = 0.05   # P5
ROBUST_HIGH_Q = 0.95   # P95


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _normalize_signal(s: str) -> str:
    if not isinstance(s, str):
        return "Unbekannt"
    low = s.strip().lower()
    if "sin" in low:
        return "Sinus"
    if "recht" in low:
        return "Rechteck"
    if "ruhe" in low:
        return "Ruhe"
    return s.strip()

def _extract_camera_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    
    m = re.search(r"(?:^|[^a-z])cam\s*[_-]?\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"Cam{int(m.group(1))}"
    return None

def _ensure_camera_column(df: pd.DataFrame) -> pd.DataFrame:
    
    cam_col = _find_col(df, ["camera", "cam", "kamera"])
    if cam_col is not None:
        df = df.copy()
        df["camera"] = df[cam_col].astype(str)
        return df

    folder_col = _find_col(df, ["folder", "ordner", "parent", "dir"])
    video_col  = _find_col(df, ["video", "filename", "datei", "file"])

    df = df.copy()
    cams = []
    for i in range(len(df)):
        cam = None
        if folder_col is not None:
            cam = _extract_camera_from_text(str(df.loc[df.index[i], folder_col]))
        if cam is None and video_col is not None:
            cam = _extract_camera_from_text(str(df.loc[df.index[i], video_col]))
        cams.append(cam if cam is not None else "Unknown")
    df["camera"] = cams
    return df

def _prepare_df(path: Path, group_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")

    df = pd.read_csv(path)

    freq_col = _find_col(df, ["frequency_hz", "freq_hz", "frequency", "freq"])
    sig_col  = _find_col(df, ["signal", "anregung", "excitation"])
    if freq_col is None:
        raise ValueError(f"{path.name}: keine Frequenzspalte gefunden (z.B. 'frequency_hz').")
    if sig_col is None:
       
        video_col = _find_col(df, ["video", "filename", "datei", "file"])
        if video_col is None:
            raise ValueError(f"{path.name}: keine 'signal' Spalte und keine 'video' Spalte zum Ableiten.")
        sig = []
        for v in df[video_col].astype(str):
            low = v.lower()
            if "sinus" in low:
                sig.append("Sinus")
            elif "rechteck" in low:
                sig.append("Rechteck")
            elif "ruhe" in low:
                sig.append("Ruhe")
            else:
                sig.append("Unbekannt")
        df = df.copy()
        df["signal"] = sig
        sig_col = "signal"

    if MAIN_COL not in df.columns:
        raise ValueError(f"{path.name}: MAIN_COL='{MAIN_COL}' nicht vorhanden. Verfügbare Spalten: {list(df.columns)}")

    df = df.copy()
    df["group"] = group_label
    df["frequency_hz"] = pd.to_numeric(df[freq_col], errors="coerce")
    df["signal"] = df[sig_col].astype(str).map(_normalize_signal)
    df[MAIN_COL] = pd.to_numeric(df[MAIN_COL], errors="coerce")

    df = _ensure_camera_column(df)

    df = df[df["signal"].isin(["Sinus", "Rechteck"])].copy()

    # drop bad rows
    df = df.dropna(subset=["frequency_hz", MAIN_COL])
    df["frequency_hz"] = df["frequency_hz"].astype(int)

    return df

def _add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """
    rel = metric / metric(ref_freq) je (group, camera, signal)
    ref_freq:
      - wenn REF_FREQ_HZ gesetzt und vorhanden -> REF_FREQ_HZ
      - sonst kleinste vorhandene Frequenz je Gruppe/Kamera/Signal
    """
    df = df.copy()

    rel_vals = []
    ref_freq_used = []

    for (g, cam, sig), sub in df.groupby(["group", "camera", "signal"], sort=False):
        sub = sub.sort_values("frequency_hz").copy()

        # Referenzfrequenz 
        if REF_FREQ_HZ is not None and (sub["frequency_hz"] == REF_FREQ_HZ).any():
            ref_f = REF_FREQ_HZ
        else:
            ref_f = int(sub["frequency_hz"].min())

        ref_row = sub[sub["frequency_hz"] == ref_f]
        ref_val = float(ref_row.iloc[0][MAIN_COL])

        if ref_val == 0 or np.isnan(ref_val):
            sub["rel_metric"] = np.nan
        else:
            sub["rel_metric"] = sub[MAIN_COL] / ref_val

        rel_vals.append(sub[["group", "camera", "signal", "frequency_hz", MAIN_COL, "rel_metric"]])
        ref_freq_used.append(pd.DataFrame({
            "group": [g], "camera": [cam], "signal": [sig], "ref_frequency_hz": [ref_f], "ref_value": [ref_val]
        }))

    out = pd.concat(rel_vals, ignore_index=True)
    refs = pd.concat(ref_freq_used, ignore_index=True)
    out = out.merge(refs, on=["group", "camera", "signal"], how="left")
    return out

def _per_camera_stats(df_rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, cam, sig), sub in df_rel.groupby(["group", "camera", "signal"], sort=False):
        v = sub["rel_metric"].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size < 2:
            continue

        q_low  = float(np.quantile(v, ROBUST_LOW_Q))
        q_high = float(np.quantile(v, ROBUST_HIGH_Q))

        rows.append({
            "group": g,
            "camera": cam,
            "signal": sig,
            "n_freq_points": int(sub["frequency_hz"].nunique()),
            "freq_min_hz": int(sub["frequency_hz"].min()),
            "freq_max_hz": int(sub["frequency_hz"].max()),
            "ref_frequency_hz": int(sub["ref_frequency_hz"].iloc[0]),
            "rel_min": float(np.min(v)),
            "rel_max": float(np.max(v)),
            "rel_p05": q_low,
            "rel_p95": q_high,
            "delta_max_minus_min": float(np.max(v) - np.min(v)),
            "delta_p95_minus_p05": float(q_high - q_low),  # robust range 
            "auc_mean_rel": float(np.mean(v)),             # einfacher AUC-Ersatz (Mittelwert über Frequenzen)
            "auc_median_rel": float(np.median(v)),
        })
    return pd.DataFrame(rows)

def _group_summary(per_cam: pd.DataFrame) -> pd.DataFrame:
    """
    Gruppiert Alt/Neu je Signal und fasst Kennzahlen über Kameras zusammen.
    """
    if per_cam.empty:
        return per_cam

    rows = []
    metrics = ["delta_p95_minus_p05", "delta_max_minus_min", "auc_mean_rel", "auc_median_rel"]
    for (g, sig), sub in per_cam.groupby(["group", "signal"], sort=False):
        row = {"group": g, "signal": sig, "n_cameras": int(sub["camera"].nunique())}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_median"] = np.nan
                row[f"{m}_std"] = np.nan
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_median"] = float(np.median(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=0))
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_alt = _prepare_df(CSV_ALT, "ALT")
    df_neu = _prepare_df(CSV_NEU, "NEU")

    df_all = pd.concat([df_alt, df_neu], ignore_index=True)

    # relative Normierung je Kamera & Signal
    df_rel = _add_relative(df_all)

    # per-frequency Tabelle 
    out_rel = OUT_DIR / f"ALT_NEU_per_frequency_relative_{MAIN_COL}.csv"
    df_rel.sort_values(["group", "camera", "signal", "frequency_hz"]).to_csv(out_rel, index=False)
    print(f"[OK] per-frequency relative CSV: {out_rel}")

    # pro Kamera & Signal Kennwerte 
    df_per_cam = _per_camera_stats(df_rel)
    out_per_cam = OUT_DIR / f"ALT_NEU_table_per_camera_{MAIN_COL}.csv"
    df_per_cam.sort_values(["signal", "group", "camera"]).to_csv(out_per_cam, index=False)
    print(f"[OK] per-camera table CSV: {out_per_cam}")

    # Gruppensummary (Alt vs Neu) je Signal 
    df_group = _group_summary(df_per_cam)
    out_group = OUT_DIR / f"ALT_NEU_summary_by_group_{MAIN_COL}.csv"
    df_group.sort_values(["signal", "group"]).to_csv(out_group, index=False)
    print(f"[OK] group summary CSV: {out_group}")

    print("\n[DONE]")
    print("Wichtige Kennzahl für Alterungsvergleich ist i.d.R.: delta_p95_minus_p05 (robuste Range).")
    print(f"Output Ordner: {OUT_DIR}")
