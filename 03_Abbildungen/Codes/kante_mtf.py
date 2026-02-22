import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Slanted Edge -> ESF -> LSF -> MTF (synthetisch, folientauglich)
# =========================================================

def gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k

def gaussian_blur_2d(img, sigma):
    """Separable Gaussian blur (ohne OpenCV/SciPy)."""
    k = gaussian_kernel_1d(sigma)
    pad = len(k) // 2

    # Blur X
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, tmp)

    # Blur Y
    tmp2 = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp2)

    return out

def make_slanted_edge(h=260, w=360, angle_deg=10.0):
    """
    Synthetisches Slanted-Edge-Target:
    Eine harte Kante, leicht geneigt (angle_deg).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Kante als Gerade: x = x0 + (y - y0)*m
    y0 = h / 2.0
    x0 = w / 2.0
    m = np.tan(np.deg2rad(angle_deg))

    # signed distance (ungefähr) zur Kante: d = x - (x0 + (y - y0)*m)
    d = xx - (x0 + (yy - y0) * m)

    # ideale harte Kante
    img = (d >= 0).astype(np.float32)

    return img, (m, x0, y0)

def compute_esf_from_edge(img, line_params, bins=500):
    """
    ESF: Intensität als Funktion des Abstands zur Kante.
    Wir projizieren Pixel auf den Normalenabstand d (in Pixel),
    binning + Mittelwert ergibt ESF(d).
    """
    h, w = img.shape
    m, x0, y0 = line_params
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # signed distance zur Geraden x - (x0 + (y - y0)*m)
    d = xx - (x0 + (yy - y0) * m)

    # Nur ROI um die Kante (damit ESF nicht von großen Flächen dominiert wird)
    roi_mask = np.abs(d) < 30  # +/- 30 px um die Kante
    d_roi = d[roi_mask]
    i_roi = img[roi_mask]

    # Binning
    dmin, dmax = np.percentile(d_roi, [1, 99])
    edges = np.linspace(dmin, dmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    esf = np.zeros(bins, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.int32)

    idx = np.searchsorted(edges, d_roi, side="right") - 1
    valid = (idx >= 0) & (idx < bins)
    idx = idx[valid]
    vals = i_roi[valid]

    # Mittelwert je Bin
    for k, v in zip(idx, vals):
        esf[k] += v
        counts[k] += 1

    # Vermeide division by zero
    counts = np.maximum(counts, 1)
    esf = esf / counts

    # -----------------------------------------------------
    # Glättung OHNE Rand-Artefakte (kein Dip am Ende)
    # Zero-padding bei np.convolve(..., mode="same") erzeugt
    # oft Dips am Rand -> wir pad-ded mit 'edge' und nutzen 'valid'
    # -----------------------------------------------------
    smooth_k = gaussian_kernel_1d(sigma=1.2)
    pad = len(smooth_k) // 2
    esf_pad = np.pad(esf, (pad, pad), mode="edge")   # alternativ: mode="reflect"
    esf = np.convolve(esf_pad, smooth_k, mode="valid")

    return centers, esf

def compute_lsf(esf_x, esf_y, trim=5):
    """
    LSF = Ableitung der ESF.
    Optional trim: schneidet Randbereiche ab (numerisch stabiler, folientauglicher).
    """
    dx = esf_x[1] - esf_x[0]
    lsf = np.gradient(esf_y, dx)

    if trim > 0:
        esf_x = esf_x[trim:-trim]
        lsf = lsf[trim:-trim]

    return esf_x, lsf

def compute_mtf(lsf, dx):
    """
    MTF = |FFT(LSF)|, normiert auf 1 bei 0 Frequenz.
    Frequenzachse in cycles/pixel.
    """
    # Fensterung (reduziert FFT-Ringing)
    win = np.hanning(len(lsf))
    lsf_w = lsf * win

    F = np.fft.rfft(lsf_w)
    mtf = np.abs(F)

    # Normierung auf DC
    mtf /= (mtf[0] + 1e-12)

    freqs = np.fft.rfftfreq(len(lsf_w), d=dx)  # cycles/pixel
    return freqs, mtf

def save_image_only(img, path, title=None):
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def save_curve(x, y, path, title, xlabel, ylabel, xlim=None, ylim=None):
    fig = plt.figure(figsize=(6.5, 4))
    plt.plot(x, y, linewidth=2.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.7, alpha=0.5)
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    # ----------------------------
    # Parameter
    # ----------------------------
    angle_deg = 10.0     # Slanted edge angle
    blur_sigma = 1.5     # simuliert optische Unschärfe
    bins = 500           # ESF Bins
    trim_lsf = 5         # Randbereiche für LSF abschneiden (0 = aus)

    # ----------------------------
    # 1) Slanted-Edge Target
    # ----------------------------
    target, line_params = make_slanted_edge(angle_deg=angle_deg)

    # „reale Abbildung“: Kante wird leicht verschmiert
    img = gaussian_blur_2d(target, sigma=blur_sigma)

    save_image_only(img, "01_slanted_edge_target.png", title="Slanted-Edge-Target (synthetisch)")

    # ----------------------------
    # 2) ESF
    # ----------------------------
    esf_x, esf_y = compute_esf_from_edge(img, line_params, bins=bins)
    save_curve(
        esf_x, esf_y,
        "02_ESF.png",
        "Edge Spread Function (ESF)",
        "Abstand zur Kante (Pixel)",
        "Intensität",
        xlim=(esf_x.min(), esf_x.max()),
        ylim=(0, 1.05)
    )

    # ----------------------------
    # 3) LSF
    # ----------------------------
    lsf_x, lsf = compute_lsf(esf_x, esf_y, trim=trim_lsf)
    save_curve(
        lsf_x, lsf,
        "03_LSF.png",
        "Line Spread Function (LSF)  (Ableitung der ESF)",
        "Abstand zur Kante (Pixel)",
        "Ableitung",
        xlim=(lsf_x.min(), lsf_x.max())
    )

    # ----------------------------
    # 4) MTF
    # ----------------------------
    dx = esf_x[1] - esf_x[0]  # Samplingabstand ESF in Pixel
    freqs, mtf = compute_mtf(lsf, dx=dx)

    # Optional: nur bis Nyquist (0.5 cycles/pixel) anzeigen
    save_curve(
        freqs, mtf,
        "04_MTF.png",
        "MTF (aus FFT der LSF, normiert)",
        "Ortsfrequenz (cycles/pixel)",
        "MTF (Kontrast)",
        xlim=(0, 0.5),
        ylim=(0, 1.05)
    )

    # ----------------------------
    # Optional: Overview (alles in einer Grafik)
    # ----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs = axs.ravel()

    axs[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Slanted-Edge-Target")
    axs[0].axis("off")

    axs[1].plot(esf_x, esf_y, linewidth=2.0)
    axs[1].set_title("ESF")
    axs[1].grid(True, linewidth=0.7, alpha=0.5)
    axs[1].set_ylim(0, 1.05)

    axs[2].plot(lsf_x, lsf, linewidth=2.0)
    axs[2].set_title("LSF (Ableitung)")
    axs[2].grid(True, linewidth=0.7, alpha=0.5)

    axs[3].plot(freqs, mtf, linewidth=2.0)
    axs[3].set_title("MTF (FFT der LSF)")
    axs[3].grid(True, linewidth=0.7, alpha=0.5)
    axs[3].set_xlim(0, 0.5)
    axs[3].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("00_overview_all.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Gespeichert:")
    print("  01_slanted_edge_target.png")
    print("  02_ESF.png")
    print("  03_LSF.png")
    print("  04_MTF.png")
    print("  00_overview_all.png (optional)")

if __name__ == "__main__":
    main()
