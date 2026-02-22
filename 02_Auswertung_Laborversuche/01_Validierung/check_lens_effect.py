#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt 


base_dir = Path(r"C:/Users/LASEHER/bachelor/data/Hoehen")

positions = ["Unten", "Mitte", "Oben"]

exts = [".png"] 

def list_images(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())

def var_laplacian(img_path: Path, roi=None):

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Bild konnte nicht gelesen werden: {img_path}")
    if roi:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    if not base_dir.exists():
        print(f"⚠️ Basisordner nicht gefunden: {base_dir.resolve()}")
        return

    results = []   
    results_t = [] 
    failures = []

    ROI = None  

    for pos in positions:
        folder = base_dir / pos
        if not folder.exists():
            print(f"Ordner nicht gefunden: {folder}")
            continue

        files = list_images(folder)
        if not files:
            print(f"Keine Bilder in {folder} (erlaubte Endungen: {', '.join(exts)})")
            continue

        vals_lap, vals_ten = [], []
        for img_file in files:
            try:
                s_lap = var_laplacian(img_file, roi=ROI)
                vals_lap.append(s_lap)


                print(f"{pos:>5} | {img_file.name:25} | Lap={s_lap:.3f}")   
 
            except Exception as e:
                failures.append((pos, img_file.name, str(e)))
                print(f"❌ Fehler bei {img_file}: {e}")

        if vals_lap:
            results.append((pos, float(np.mean(vals_lap)), vals_lap))
        if vals_ten:
            results_t.append((pos, float(np.mean(vals_ten)), vals_ten))

    # CSV 1: einfache Ergebnisse
    csv_path = base_dir / "schaerfe_ergebnisse.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Position", "Mittelwert_VarLaplacian", "Einzelwerte"])
        for pos, mean_val, vals in results:
            writer.writerow([pos, f"{mean_val:.6f}", ", ".join(f"{v:.6f}" for v in vals)])
    print(f"\n✅ Ergebnisse (Laplacian) gespeichert in: {csv_path.resolve()}")

    # CSV 2: Statistik & Normierung (zu 'Mitte')
    ref_name = "Mitte"
    ref_mean = next((m for (pos, m, _) in results if pos == ref_name), None)

    csv2_path = base_dir / "schaerfe_ergebnisse_stat_norm.csv"
    with open(csv2_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "Position",
            "Mittelwert_VarLaplacian",
            "StdAbw",
            "Min",
            "Max",
            "Norm_zu_Mitte",
            "Delta_%_zu_Mitte",
            "Einzelwerte"
        ])
        for pos, mean_val, vals in results:
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if ref_mean and ref_mean != 0:
                norm = mean_val / ref_mean
                delta_pct = (norm - 1.0) * 100.0
            else:
                norm = math.nan
                delta_pct = math.nan
            writer.writerow([
                pos,
                f"{mean_val:.6f}",
                f"{std:.6f}",
                f"{vmin:.6f}",
                f"{vmax:.6f}",
                f"{norm:.6f}",
                f"{delta_pct:.3f}",
                ", ".join(f"{v:.6f}" for v in vals)
            ])
    print(f"📄 Statistik & Normierung gespeichert in: {csv2_path.resolve()}")

    if results:
      
        means = {pos: m for (pos, m, _) in results}

        labels = [p for p in positions if p in means]
        values = [means[p] for p in labels]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color="#66b3ff", width=0.5) 
        plt.ylabel("Schärfe (Varianz Laplacian)")
        plt.xlabel("Position")
        plt.title("Durchschnittliche Bildschärfe je Position")

        ymin = min(values) * 0.95
        ymax = max(values) * 1.05
        plt.ylim(ymin, ymax)

    
        if "Mitte" in means:
            plt.axhline(y=means["Mitte"], color="black", linestyle="-", linewidth=1, label="Mitte")
            plt.legend()

        plt.tight_layout()
        plot_path = base_dir / "sharpness_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"🖼️ Plot gespeichert: {plot_path.resolve()}")

    if failures:
        log_path = base_dir / "lesefehler_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            for pos, name, err in failures:
                f.write(f"[{pos}] {name} :: {err}\n")
        print(f"⚠️ Es gab Lesefehler. Details in: {log_path.resolve()}")

if __name__ == "__main__":
    main()
