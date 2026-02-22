import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Parameters (edit these)
# -----------------------------
EXPOSURE_MS = 10.0      # e.g. 10 ms
FPS = 30.0              # e.g. 30 fps
NYQUIST = FPS / 2.0
A_SHIFT_PX = 3.0        # illustrative motion amplitude in "px"
F_BLUR_HZ = 8.0         # example frequency for blur illustration (can be in your 2–30 Hz range)

# Controls how "rounded" the BLURRED edge is (in px).
EDGE_SIGMA_PX = 0.6

OUT_BLUR = "01_motion_blur_exposure_simple.png"
OUT_ALIAS = "02_sampling_aliasing_simple.png"


# -----------------------------
# 1) Motion blur: exposure integration explained clearly
# -----------------------------
def make_motion_blur_exposure_simple(exposure_ms=10.0, f_hz=8.0, A_px=3.0,
                                     edge_sigma_px=0.6, out_path=OUT_BLUR):
    """
    Two-part schematic:
      Top: displacement over time within exposure.
      Bottom: ideal sharp edge vs averaged (blurred) edge due to shifting during exposure.
    """
    T = exposure_ms / 1000.0
    t = np.linspace(0, T, 600)
    shift = A_px * np.sin(2 * np.pi * f_hz * t)

    x = np.linspace(-25, 25, 2500)

    # Ideal (perfect) edge: hard step
    def sharp_edge(x0):
        return (x >= x0).astype(float)

    # Realistic edge model for the BLUR averaging: smooth transition
    def soft_edge(x0):
        sigma = max(1e-6, float(edge_sigma_px))
        return 1.0 / (1.0 + np.exp(-(x - x0) / sigma))

    # Blur: average of multiple shifted *soft* edges over the exposure
    sample_idx = np.linspace(0, len(t) - 1, 40).astype(int)
    I_blur = np.zeros_like(x, dtype=float)
    for s in shift[sample_idx]:
        I_blur += soft_edge(s)
    I_blur /= len(sample_idx)

    # Sharp reference: perfect step (not curved)
    I_sharp = sharp_edge(0.0)

    fig = plt.figure(figsize=(12, 6), dpi=160)

    # --- Top: motion during exposure
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t * 1000.0, shift)
    ax1.set_title("Bewegung während der Belichtung → zeitliche Integration", fontsize=14)
    ax1.set_xlabel("Zeit innerhalb der Belichtung [ms]")
    ax1.set_ylabel("Bildverschiebung (schematisch) [px]")
    ax1.grid(True, alpha=0.3)

    y_min, y_max = ax1.get_ylim()
    ax1.add_patch(Rectangle((0, y_min), exposure_ms, y_max - y_min, fill=False, linewidth=2))

    # --- Bottom: sharp vs blurred edge
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, I_sharp, label="Ideale Kante (Kamera ruhig)")
    ax2.plot(
        x, I_blur,
        color="#B388EB",
        linewidth=2,
        label="Gemittelte Kante (Kamera bewegt sich während Belichtung)"
    )

    ax2.set_title("Was im Bild passiert: viele leicht verschobene Kanten werden gemittelt", fontsize=13)
    ax2.set_xlabel("Position [px]")
    ax2.set_ylabel("Intensität (0…1)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 2) Sampling & aliasing
# -----------------------------
def make_sampling_aliasing_simple(fps=30.0, f_low=10.0, f_high=20.0, A=1.0, duration_s=0.9, out_path=OUT_ALIAS):
    nyq = fps / 2.0
    dt = 1.0 / fps

    def make_signals(f_hz):
        t_cont = np.linspace(0, duration_s, 3000)
        x_cont = A * np.sin(2 * np.pi * f_hz * t_cont)
        t_s = np.arange(0, duration_s + 1e-12, dt)
        x_s = A * np.sin(2 * np.pi * f_hz * t_s)

        k = int(np.round(f_hz / fps))
        f_alias = abs(f_hz - k * fps)
        return t_cont, x_cont, t_s, x_s, f_alias

    fig = plt.figure(figsize=(14, 6), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.18)

    cases = [f_low, f_high]

    for i, f_hz in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        t_cont, x_cont, t_s, x_s, f_alias = make_signals(f_hz)

        ax.plot(t_cont, x_cont, linewidth=2, label="Echte Bewegung (kontinuierlich)")
        ax.plot(
            t_s, x_s,
            marker="o",
            linewidth=1.5,
            color="#B388EB",
            label=f"Frames ({fps:.0f} fps)"
        )

        ax.set_title(f"{f_hz:.0f} Hz", fontsize=13)
        ax.set_xlabel("Zeit [s]")
        ax.set_ylabel("Bewegung (schematisch)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-1.35, 1.35)

        ax.text(
            0.02, 0.98,
            f"Nyquist-Grenze bei {fps:.0f} fps: {nyq:.0f} Hz\n"
            f"Aliasing: {f_alias:.0f} Hz",
            transform=ax.transAxes, va="top", ha="left", fontsize=10
        )

        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Zeitliche Abtastung (Frames) & Aliasing", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    make_motion_blur_exposure_simple(
        EXPOSURE_MS,
        f_hz=F_BLUR_HZ,
        A_px=A_SHIFT_PX,
        edge_sigma_px=EDGE_SIGMA_PX,
        out_path=OUT_BLUR
    )
    make_sampling_aliasing_simple(FPS, f_low=10.0, f_high=20.0, A=1.0, duration_s=0.5, out_path=OUT_ALIAS)

    print("Saved:")
    print(" -", OUT_BLUR)
    print(" -", OUT_ALIAS)
