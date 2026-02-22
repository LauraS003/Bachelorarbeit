import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =========================================================
# Beispielbild + Laplace-Antwort (mehr als nur eine Kante)
# - Erzeugt synthetisches Testbild mit mehreren Strukturen
# - Berechnet Laplace-Antwort per 3x3 Kernel (4-Nachbarschaft)
# - Speichert eine Paper-taugliche Abbildung
# =========================================================

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(region * kernel)
    return out

def make_test_image(h=240, w=360) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)

    # (A) sanfter Hintergrund-Gradient (zeigt: Laplace ~0 bei linearem Verlauf)
    x = np.linspace(0, 1, w, dtype=np.float32)
    img += 0.15 * x[None, :]

    # (B) gefülltes Quadrat
    img[60:140, 50:130] = 0.85

    # (C) Rechteck-Rahmen (dünn)
    img[40:42, 200:320] = 1.0
    img[160:162, 200:320] = 1.0
    img[40:162, 200:202] = 1.0
    img[40:162, 318:320] = 1.0

    # (D) Kreis (gefüllt)
    cy, cx, r = 140, 160, 35
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    img[mask] = 0.65

    # (E) dünne diagonale Linie
    for i in range(30, 190):
        j = int(0.9 * i)  # Steigung
        if 0 <= j < w:
            img[i:i+2, j:j+2] = 1.0

    return np.clip(img, 0, 1)

def main():
    # 1) Testbild erzeugen
    img = make_test_image()

    # 2) Laplace-Operator (diskret)
    laplace_kernel = np.array([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]], dtype=np.float32)

    lap = convolve2d(img, laplace_kernel)

    # 3) Darstellungen vorbereiten
    # Betrag für "Kantenstärke" (so sieht man die weißen Linien wie auf Folien oft üblich)
    lap_abs = np.abs(lap)
    lap_abs = lap_abs / (lap_abs.max() + 1e-9)

    # Vorzeichenbehaftet (optional, wissenschaftlich sauber)
    lap_signed = lap / (np.max(np.abs(lap)) + 1e-9)

    # 4) Plot: Bild + |Laplace| + (optional) signed Laplace
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, wspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Eingabebild")
    ax0.axis("off")

    ax1.imshow(lap_abs, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("|Laplace-Antwort| (Kantenstärke)")
    ax1.axis("off")

    # signed Laplace mit diverging colormap (zeigt + / - Peaks)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im2 = ax2.imshow(lap_signed, cmap="seismic", norm=norm)
    ax2.set_title("Laplace-Antwort (mit Vorzeichen)")
    ax2.axis("off")

    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar.set_label("negativ  ←  0  →  positiv")

    plt.tight_layout()

    # 5) Speichern
    out_path = "laplace_beispiel_mehr_struktur.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Gespeichert:", out_path)

    # Optional: einzelne Bilder separat speichern (für PowerPoint-Einbau)
    plt.imsave("beispielbild.png", img, cmap="gray", vmin=0, vmax=1)
    plt.imsave("laplace_abs.png", lap_abs, cmap="gray", vmin=0, vmax=1)
    # für signed separat speichern:
    # (als PNG mit colormap ist ok, aber Werte sind dann nicht mehr numerisch)
    plt.imsave("laplace_signed.png", lap_signed, cmap="seismic", vmin=-1, vmax=1)

if __name__ == "__main__":
    main()
