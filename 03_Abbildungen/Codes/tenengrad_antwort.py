import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# =========================================================
# Eingabebild (identisch zum Laplace-Beispiel)
# =========================================================

def make_test_image(h=240, w=360) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)

    # Sanfter Hintergrundgradient
    x = np.linspace(0, 1, w, dtype=np.float32)
    img += 0.15 * x[None, :]

    # Gefülltes Quadrat
    img[60:140, 50:130] = 0.85

    # Rechteck-Rahmen
    img[40:42, 200:320] = 1.0
    img[160:162, 200:320] = 1.0
    img[40:162, 200:202] = 1.0
    img[40:162, 318:320] = 1.0

    # Kreis
    cy, cx, r = 140, 160, 35
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    img[mask] = 0.65

    # Diagonale Linie
    for i in range(30, 190):
        j = int(0.9 * i)
        if 0 <= j < w:
            img[i:i+2, j:j+2] = 1.0

    return np.clip(img, 0, 1)


# =========================================================
# Sobel-Kernel (Standard)
# =========================================================

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]], dtype=np.float32)


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = np.sum(
                padded[y:y+kh, x:x+kw] * kernel
            )
    return out


# =========================================================
# Hauptteil
# =========================================================

img = make_test_image()

# Sobel-Ableitungen
Gx = convolve2d(img, SOBEL_X)
Gy = convolve2d(img, SOBEL_Y)

# Tenengrad / Gradientenenergie
tenengrad = Gx**2 + Gy**2

# Normierung für Darstellung
def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

img_d = norm01(img)
Gx_d = norm01(np.abs(Gx))
Gy_d = norm01(np.abs(Gy))
T_d  = norm01(tenengrad)

# =========================================================
# Darstellung (folientauglich)
# =========================================================

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs = axs.ravel()

axs[0].imshow(img_d, cmap="gray")
axs[0].set_title("Eingabebild")
axs[0].axis("off")

axs[1].imshow(Gx_d, cmap="gray")
axs[1].set_title(r"Sobel $|G_x|$")
axs[1].axis("off")

axs[2].imshow(Gy_d, cmap="gray")
axs[2].set_title(r"Sobel $|G_y|$")
axs[2].axis("off")

axs[3].imshow(T_d, cmap="gray")
axs[3].set_title(r"Gradientenenergie (Tenengrad) $G_x^2 + G_y^2$")
axs[3].axis("off")

plt.tight_layout()
plt.savefig("tenengrad_beispiel_gleiches_eingabebild.png", dpi=300, bbox_inches="tight")
plt.show()

print("Gespeichert: tenengrad_beispiel_gleiches_eingabebild.png")
