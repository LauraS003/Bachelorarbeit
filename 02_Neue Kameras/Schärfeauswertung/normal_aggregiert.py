import os
import pandas as pd
import matplotlib.pyplot as plt


laplace_csv = r"D:\BA\Messungen\Frequenzen\Videos\normal_Auswertung_Laplace\_Combined\laplacian_summary_all_cams.csv"
tenengrad_csv = r"D:\BA\Messungen\Frequenzen\Videos\normal_Auswertung_Tenengrad\_Combined\tenengrad_summary_all_cams.csv"

out_dir = os.path.dirname(laplace_csv)

def drop_and_rename_cams(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Entfernt Cam1 (defekt)
    - Benennt Cam2->Cam1, Cam3->Cam2, Cam4->Cam3 um
    """
    df = df.copy()

    # Drop defekte Cam1
    df = df[df["camera"].astype(str).str.lower() != "cam1"].copy()

    # Rename mapping
    mapping = {"Cam2": "Cam1", "Cam3": "Cam2", "Cam4": "Cam3",
               "cam2": "Cam1", "cam3": "Cam2", "cam4": "Cam3"}
    df["camera"] = df["camera"].apply(lambda x: mapping.get(str(x), str(x)))

    return df


def build_aggregated_relative_curve(
    df: pd.DataFrame,
    median_col: str,
    ref_freq: float = 2.0
) -> pd.DataFrame:
   
    df = df.copy()

    df["frequency_hz"] = pd.to_numeric(df["frequency_hz"], errors="coerce")

    ref = df[df["frequency_hz"] == ref_freq][["camera", "signal", median_col]].copy()
    ref = ref.rename(columns={median_col: "ref_value"})

    df = df.merge(ref, on=["camera", "signal"], how="left")

    df = df.dropna(subset=["ref_value"])
    df["rel_value"] = df[median_col] / df["ref_value"]

    # Aggregation über Kameras
    agg = (
        df.groupby(["frequency_hz", "signal"])["rel_value"]
        .median()
        .reset_index()
        .sort_values("frequency_hz")
    )

    return agg


def plot_agg(agg: pd.DataFrame, title: str, y_label: str, out_path: str) -> None:
   
    plt.figure(figsize=(10.5, 6))
    for sig in ["Sinus", "Rechteck"]:
        sub = agg[agg["signal"] == sig]
        if sub.empty:
            continue
        plt.plot(sub["frequency_hz"], sub["rel_value"], marker="o", label=sig)

    plt.title(title)
    plt.xlabel("Anregungsfrequenz [Hz]")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Laplace
    df_lap = pd.read_csv(laplace_csv)
    df_lap = drop_and_rename_cams(df_lap)

    lap_median_col = "median_laplacian"
    if lap_median_col not in df_lap.columns:
        raise ValueError(f"Spalte '{lap_median_col}' nicht gefunden. Verfügbare Spalten: {list(df_lap.columns)}")

    agg_lap = build_aggregated_relative_curve(df_lap, median_col=lap_median_col, ref_freq=2.0)

    out_lap = os.path.join(out_dir, "AGG_relative_sharpness_laplacian.png")
    plot_agg(
        agg_lap,
        title="Aggregierter relativer Schärfeverlauf (Laplace, Median über Kameras)",
        y_label="Normierte Laplace-Varianz [-] (Median/Referenz @ 2 Hz)",
        out_path=out_lap
    )

    # Tenengrad
    df_ten = pd.read_csv(tenengrad_csv)
    df_ten = drop_and_rename_cams(df_ten)

    ten_median_col = "median_tenengrad"
    if ten_median_col not in df_ten.columns:
    
        candidates = [c for c in df_ten.columns if c.lower().startswith("median_")]
        raise ValueError(
            f"Spalte '{ten_median_col}' nicht gefunden. Mögliche Median-Spalten: {candidates}"
        )

    agg_ten = build_aggregated_relative_curve(df_ten, median_col=ten_median_col, ref_freq=2.0)

    out_ten = os.path.join(out_dir, "AGG_relative_sharpness_tenengrad.png")
    plot_agg(
        agg_ten,
        title="Aggregierter relativer Schärfeverlauf (Tenengrad, Median über Kameras)",
        y_label="Normierter Tenengrad [-] (Median/Referenz @ 2 Hz)",
        out_path=out_ten
    )

    agg_out = agg_lap.merge(
        agg_ten, on=["frequency_hz", "signal"], how="outer", suffixes=("_laplace", "_tenengrad")
    )
    agg_csv_path = os.path.join(out_dir, "AGG_relative_sharpness_laplace_tenengrad.csv")
    agg_out.to_csv(agg_csv_path, index=False)

    print("Fertig! Gespeichert:")
    print(" -", out_lap)
    print(" -", out_ten)
    print(" -", agg_csv_path)


if __name__ == "__main__":
    main()
