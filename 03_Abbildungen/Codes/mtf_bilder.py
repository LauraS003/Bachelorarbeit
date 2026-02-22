import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k

def gaussian_blur_2d(img, sigma):
    """Separable Gaussian blur without OpenCV."""
    k = gaussian_kernel_1d(sigma)
    # blur x
    pad = len(k) // 2
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, tmp)
    # blur y
    tmp2 = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp2)
    return out

def contrast_rms(img):
    """RMS contrast (std dev) as a simple contrast measure."""
    return float(np.std(img))

# ---------------------------------------------------------
# A) Sinus patterns + blur (Ortsfrequenz vs Kontrastübertragung)
# ---------------------------------------------------------
H, W = 120, 360
y = np.linspace(0, 1, H)[:, None]
x = np.linspace(0, 1, W)[None, :]

freqs = [2, 6, 14]  # low -> mid -> high spatial frequency (cycles across width)
patterns = []
blurred = []
sigma = 2.0  # "optical blur strength"

for f in freqs:
    # sinus in [0,1]
    pat = 0.5 + 0.5 * np.sin(2 * np.pi * f * x)
    pat = np.repeat(pat, H, axis=0).astype(np.float32)
    patterns.append(pat)
    blurred.append(gaussian_blur_2d(pat, sigma=sigma))

# Combine into a single image: original row / blurred row
fig, axs = plt.subplots(2, 3, figsize=(11, 3.8))
for i, f in enumerate(freqs):
    axs[0, i].imshow(patterns[i], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axs[0, i].set_title(f"Ortsfrequenz: {f} (niedrig→hoch)")
    axs[0, i].axis("off")

    axs[1, i].imshow(blurred[i], cmap="gray", vmin=0, vmax=1, aspect="auto")
    axs[1, i].set_title("nach System (Blur)")
    axs[1, i].axis("off")

plt.tight_layout()
plt.savefig("mtf_grundidee_sinus_blur.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# Contrast transfer over spatial frequency (simple "MTF-like" plot)
# Using RMS contrast ratio: C_out / C_in
# ---------------------------------------------------------
ratios = []
for i in range(len(freqs)):
    Cin = contrast_rms(patterns[i])
    Cout = contrast_rms(blurred[i])
    ratios.append(Cout / (Cin + 1e-12))

fig = plt.figure(figsize=(6.5, 4.0))
plt.plot(freqs, ratios, linewidth=2.0, marker="o")
plt.ylim(0, 1.05)
plt.xlabel("Ortsfrequenz (relativ)")
plt.ylabel("Kontrastübertragung (C_out / C_in)")
plt.title("Kontrastübertragung nimmt mit Ortsfrequenz ab")
plt.grid(True, linewidth=0.7, alpha=0.5)
plt.tight_layout()
plt.savefig("mtf_grundidee_kontrast_vs_freq.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------
# B) Idealized MTF curve (clean, slide-friendly)
# ---------------------------------------------------------
f = np.linspace(0, 1, 400)
# A plausible smooth falloff (purely illustrative)
mtf = np.exp(- (f / 0.35)**1.6)

fig = plt.figure(figsize=(6.5, 4.0))
plt.plot(f, mtf, linewidth=2.0)
plt.ylim(0, 1.05)
plt.xlabel("Ortsfrequenz (normiert)")
plt.ylabel("MTF (Kontrast)")
plt.title("MTF-Kurve (schematisch)")
plt.grid(True, linewidth=0.7, alpha=0.5)

# Optional: mark MTF50
idx = np.argmin(np.abs(mtf - 0.5))
plt.scatter([f[idx]], [mtf[idx]])
plt.text(f[idx] + 0.02, 0.52, "MTF50", fontsize=10)

plt.tight_layout()
plt.savefig("mtf_grundidee_mtf_kurve.png", dpi=300, bbox_inches="tight")
plt.show()

print("Gespeichert: mtf_grundidee_sinus_blur.png")
print("Gespeichert: mtf_grundidee_kontrast_vs_freq.png")
print("Gespeichert: mtf_grundidee_mtf_kurve.png")
