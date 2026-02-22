from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path(r"D:\BA\Messungen\Frequenzen\MTF\_Auswertung_MTF\mtf_summary_FIXED_TO_RUHE_WITH_MTF_AT.csv")

OUT_DIR = CSV_PATH.parent
OUT_FIG = OUT_DIR / "Figure_MTF50_vs_MTFat005_vs_Std.png"


def pick_first_existing(df, candidates):
    """Return first column name that exists, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_df(df):
    df = df.copy()

    df["frequency_hz_num"] = pd.to_numeric(df.get("frequency_hz", None), errors="coerce")

    if "signal" in df.columns:
        df = df[df["signal"].isin(["Sinus", "Rechteck", "Ruhe"])].copy()

    return df

def plot_line(ax, sub, xcol, ycol, label, style):
    sub = sub.sort_values(xcol)
    ax.plot(sub[xcol], sub[ycol], style, label=label)

df = pd.read_csv(CSV_PATH)
df = prepare_df(df)

mtf50_col = pick_first_existing(df, ["mtf50_median", "mtf50_mean", "mtf50_ref"])
mtf005_col = pick_first_existing(df, ["mtf_at_005_median", "mtf_at_005_mean", "mtf_at_005_ref"])
std005_col = pick_first_existing(df, ["mtf_at_005_std"])

if mtf50_col is None:
    raise RuntimeError("Keine passende MTF50-Spalte gefunden (erwartet: mtf50_median/mean/ref).")
if mtf005_col is None:
    raise RuntimeError("Keine passende MTF@0.05-Spalte gefunden (erwartet: mtf_at_005_median/mean/ref).")
if std005_col is None:
    raise RuntimeError("Keine passende Std-Spalte gefunden (erwartet: mtf_at_005_std).")

df_freq = df[~df["frequency_hz_num"].isna()].copy()

fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.5), sharex=True)

ax = axes[0]
for signal, style in [("Sinus", "-o"), ("Rechteck", "-s")]:
    sub = df_freq[df_freq["signal"] == signal]
    if not sub.empty:
        plot_line(ax, sub, "frequency_hz_num", mtf50_col, signal, style)

ax.set_ylabel("MTF50 [cy/px]")
ax.set_title("MTF50 vs. Anregungsfrequenz")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
for signal, style in [("Sinus", "-o"), ("Rechteck", "-s")]:
    sub = df_freq[df_freq["signal"] == signal]
    if not sub.empty:
        plot_line(ax, sub, "frequency_hz_num", mtf005_col, signal, style)

ax.set_ylabel("MTF@0.05 [-]")
ax.set_title("MTF@0.05 cy/px vs. Anregungsfrequenz")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[2]
for signal, style in [("Sinus", "-o"), ("Rechteck", "-s")]:
    sub = df_freq[df_freq["signal"] == signal]
    if not sub.empty:
        plot_line(ax, sub, "frequency_hz_num", std005_col, signal, style)

ax.set_xlabel("Anregungsfrequenz [Hz]")
ax.set_ylabel("Std(MTF@0.05) [-]")
ax.set_title("Zeitliche Instabilität: Std(MTF@0.05) vs. Anregungsfrequenz")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"[OK] Saved: {OUT_FIG}")
print(f"[INFO] Used columns: MTF50='{mtf50_col}', MTF@0.05='{mtf005_col}', Std@0.05='{std005_col}'")
