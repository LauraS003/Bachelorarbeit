from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Alt-Zusammenfassung (gealterte Kameras) aus Laplace-Skript
ALT_SUMMARY_CSV = Path(r"D:\BA\Messungen\Alt\Frequenz_Videos\_Auswertung_Laplace\laplacian_summary_alt.csv")

# Output-Ordner für neue Plots
OUT_DIR = ALT_SUMMARY_CSV.parent / "_Plots_pro_Kamera"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hauptkennwert
USE_MEDIAN = True  # True => median_laplacian, False => mean_laplacian
VALUE_COL = "median_laplacian" if USE_MEDIAN else "mean_laplacian"

# normalisieren je Kamera auf Ruhe
MAKE_REL_TO_REST = True

# Plot settings
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


def must_have_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV fehlt Spalten: {missing}\nVorhanden: {list(df.columns)}")


def main():
    if not ALT_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Nicht gefunden: {ALT_SUMMARY_CSV}")

    df = pd.read_csv(ALT_SUMMARY_CSV)

    must_have_cols(df, ["camera", "signal", "frequency_hz", VALUE_COL])

    # Frequenz numerisch, Ruhe raus für Frequenzplots
    df["frequency_hz_num"] = pd.to_numeric(df["frequency_hz"], errors="coerce")

    # Kamera-Liste (nur gealterte)
    cams = sorted([c for c in df["camera"].dropna().unique() if str(c).lower().startswith("cam_")])
    if not cams:
        print("[WARN] Keine Cam_* gefunden. Prüfe Spalte 'camera' in der CSV.")
        print("Gefundene cameras:", sorted(df["camera"].dropna().unique()))
        return

    # Ruhe-Werte je Kamera
    rest_map = {}
    if MAKE_REL_TO_REST:
        rest = df[df["signal"].astype(str).str.lower() == "ruhe"].copy()
        for _, r in rest.iterrows():
            cam = str(r["camera"])
            rest_val = float(r.get(VALUE_COL, np.nan))
            if not np.isnan(rest_val) and rest_val != 0:
                rest_map[cam] = rest_val

    # Für jeden Cam_* ein Plot: Sinus + Rechteck
    for cam in cams:
        sub = df[(df["camera"] == cam) & (df["signal"].isin(["Sinus", "Rechteck"]))].copy()
        sub = sub[~sub["frequency_hz_num"].isna()].copy()
        sub = sub.sort_values(["signal", "frequency_hz_num"])

        if sub.empty:
            print(f"[WARN] Keine Frequenzdaten für {cam}.")
            continue

        # Absolut: Laplace vs Frequenz
        plt.figure(figsize=(7.8, 4.8))

        for sig, marker in [("Sinus", "o"), ("Rechteck", "s")]:
            s = sub[sub["signal"] == sig].sort_values("frequency_hz_num")
            if s.empty:
                continue
            plt.plot(
                s["frequency_hz_num"],
                s[VALUE_COL].astype(float),
                marker=marker,
                linewidth=2.0,
                label=sig
            )

        plt.xlabel("Anregungsfrequenz [Hz]")
        plt.ylabel(f"Laplacian-Varianz [-] ({'Median' if USE_MEDIAN else 'Mittelwert'})")
        plt.title(f"{cam}: Schärfeverlauf (Sinus vs. Rechteck) – absolut")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        export_plot(OUT_DIR / f"{cam}_Laplace_abs_Sinus_vs_Rechteck.png")

    
        # Relativ zu Ruhe
        if MAKE_REL_TO_REST and cam in rest_map:
            ref = rest_map[cam]
            plt.figure(figsize=(7.8, 4.8))

            for sig, marker in [("Sinus", "o"), ("Rechteck", "s")]:
                s = sub[sub["signal"] == sig].sort_values("frequency_hz_num")
                if s.empty:
                    continue
                y = s[VALUE_COL].astype(float) / float(ref)

                plt.plot(
                    s["frequency_hz_num"],
                    y,
                    marker=marker,
                    linewidth=2.0,
                    label=sig
                )

            plt.axhline(1.0, linewidth=1.0)
            plt.xlabel("Anregungsfrequenz [Hz]")
            plt.ylabel(f"Relativ zu Ruhe [-] ({'Median' if USE_MEDIAN else 'Mittelwert'})")
            plt.title(f"{cam}: Schärfeverlauf (Sinus vs. Rechteck) – relativ zu Ruhe")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            export_plot(OUT_DIR / f"{cam}_Laplace_relZuRuhe_Sinus_vs_Rechteck.png")
        else:
            if MAKE_REL_TO_REST:
                print(f"[INFO] {cam}: Kein/ungültiger Ruhe-Referenzwert gefunden -> Relativplot übersprungen.")

    print("\n[DONE] Pro Kamera wurden Sinus+Rechteck Plots erstellt.")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

