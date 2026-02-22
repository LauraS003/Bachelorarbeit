from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"D:\BA\Messungen\Frequenzen\MTF")
OUT_DIR  = BASE_DIR / "_Auswertung_MTF"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# SEARCH ROI
SEARCH_ROI = (900, 80, 2600, 2000)  

USE_GREEN_CHANNEL = True   # Bayer/Demosaic

ROI_SIZE_DEFAULT = 260  
MIN_ROI_SIZE = 160

MAX_FRAMES = 250
FRAME_STRIDE = 2
REF_FRAME_INDEX = 0

OVERSAMPLE = 4
SMOOTH_ESF = 7
WINDOW_LSF = True

NORM_BAND = (0.02, 0.10)

PLOT_XLIM = (0.0, 0.50)
PLOT_YLIM = (0.0, 1.10)

MTF_SMOOTH_KERNEL = 7  

SAVE_MTF_CURVE_IMAGE  = True
SAVE_ROIROTATED_IMAGE = True

MTF_AT_FREQS = [0.05, 0.10, 0.20]  

R_BAND_INNER = 0.25
R_BAND_OUTER = 0.78

ANGLE_STEP_DEG = 0.5
PEAK_MIN_SEP_DEG = 8.0
TOP_K_PEAKS = 8

def export_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def list_freq_dirs(base: Path):
    freq_dirs = []
    ruhe_dir = None
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if p.name.lower() == "ruhe":
            ruhe_dir = p
        elif re.match(r"^\d+Hz$", p.name):
            freq_dirs.append(p)
    freq_dirs.sort(key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    return freq_dirs, ruhe_dir

def find_video(patterns, folder):
    for pat in patterns:
        files = sorted(folder.glob(pat))
        if files:
            return files[0]
    return None

def get_frame(video, idx):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Frame konnte nicht gelesen werden")
    return frame

def iter_frames(video):
    cap = cv2.VideoCapture(str(video))
    i = 0
    used = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % FRAME_STRIDE == 0:
            yield frame
            used += 1
            if MAX_FRAMES and used >= MAX_FRAMES:
                break
        i += 1
    cap.release()

def crop_with_roi(img, roi):
    x0, y0, w, h = roi
    return img[y0:y0+h, x0:x0+w], (x0, y0)

def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if USE_GREEN_CHANNEL:
        return frame_bgr[:, :, 1].copy()
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

def estimate_circle_from_roi(gray_roi):
    g = cv2.GaussianBlur(gray_roi, (7,7), 1.2)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h,w = gray_roi.shape
        return w/2, h/2, min(w,h)*0.42
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    (cx,cy), r = cv2.minEnclosingCircle(contours[0])
    return float(cx), float(cy), float(r)

def gradient_mag(gray):
    g = cv2.GaussianBlur(gray, (5,5), 1.2)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag

def radial_energy_profile(mag, cx, cy, r):
    r0 = max(5.0, r * R_BAND_INNER)
    r1 = min(r * R_BAND_OUTER, 0.98 * r)
    n_r = int(max(80, (r1 - r0)))
    rs = np.linspace(r0, r1, n_r)

    angles = np.arange(0.0, 180.0, ANGLE_STEP_DEG)
    energy = np.zeros_like(angles, dtype=np.float32)

    h, w = mag.shape
    for i, a in enumerate(angles):
        th = np.deg2rad(a)
        xs = cx + rs * np.cos(th)
        ys = cy + rs * np.sin(th)
        xi = np.clip(np.round(xs).astype(int), 0, w-1)
        yi = np.clip(np.round(ys).astype(int), 0, h-1)
        energy[i] = float(np.mean(mag[yi, xi]))
    return angles, energy

def find_peaks_simple(x, y, min_sep_deg):
    peaks = []
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            peaks.append((float(x[i]), float(y[i])))
    peaks.sort(key=lambda t: t[1], reverse=True)
    chosen = []
    for ang, val in peaks:
        if all(abs(ang - a2) >= min_sep_deg for a2, _ in chosen):
            chosen.append((ang, val))
    return chosen

def angle_to_line_points(cx, cy, ang_deg, w, h):
    th = np.deg2rad(ang_deg)
    dx, dy = np.cos(th), np.sin(th)
    L = max(w, h) * 2
    p1 = (int(cx - dx*L), int(cy - dy*L))
    p2 = (int(cx + dx*L), int(cy + dy*L))
    return p1, p2

def extract_rotated_roi(gray, p1, p2, mx, my, roi_size):
    h, w = gray.shape
    ang = float(np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0])))
    rot = -(ang - 90.0)
    M = cv2.getRotationMatrix2D((float(mx), float(my)), rot, 1.0)
    rot_img = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)

    half = roi_size // 2
    x0 = max(int(mx - half), 0); y0 = max(int(my - half), 0)
    x1 = min(int(mx + half), w); y1 = min(int(my + half), h)
    roi = rot_img[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < MIN_ROI_SIZE or roi.shape[1] < MIN_ROI_SIZE:
        return None, None
    return roi, (x0, y0, x1, y1)

def _choose_esf_axis_from_roi(roi: np.ndarray) -> int:
    """
    Return:
        0 => ESF entlang x  (roi.mean(axis=0))  [typisch bei vertikaler Kante]
        1 => ESF entlang y  (roi.mean(axis=1))  [typisch bei horizontaler Kante]
    """
    r = roi.astype(np.float32)
    r = cv2.GaussianBlur(r, (5,5), 1.0)

    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)

    sx = float(np.mean(np.abs(gx)))
    sy = float(np.mean(np.abs(gy)))
    return 0 if sx >= sy else 1

def _crossings_down(freqs: np.ndarray, y: np.ndarray, level: float = 0.5, start_idx: int = 0):
    crossings = []
    s = max(1, start_idx + 1)
    for i in range(s, len(y)):
        if (y[i-1] > level) and (y[i] <= level):
            crossings.append((i-1, i))
    return crossings

def _interp_cross(freqs: np.ndarray, y: np.ndarray, i0: int, i1: int, level: float = 0.5) -> float:
    f0, f1 = float(freqs[i0]), float(freqs[i1])
    y0, y1 = float(y[i0]), float(y[i1])
    if y1 == y0:
        return f1
    t = (level - y0) / (y1 - y0)
    return float(f0 + t * (f1 - f0))

def _mtf_at(freqs: np.ndarray, mtf_curve: np.ndarray, f_target: float) -> float:
    if f_target < float(freqs[0]) or f_target > float(freqs[-1]):
        return float("nan")
    return float(np.interp(f_target, freqs, mtf_curve))

def mtf_from_roi_normalized(
    roi: np.ndarray,
    oversample=OVERSAMPLE,
    smooth_esf=SMOOTH_ESF,
    window_lsf=WINDOW_LSF,
    norm_band=NORM_BAND,
    mtf_smooth_kernel=MTF_SMOOTH_KERNEL,
    mtf_at_freqs=MTF_AT_FREQS
):
    
    esf_axis = _choose_esf_axis_from_roi(roi)

    # ESF
    esf = roi.mean(axis=esf_axis).astype(float)
    if smooth_esf and smooth_esf > 1:
        esf = np.convolve(esf, np.ones(smooth_esf)/smooth_esf, mode="same")

    # Oversampling
    x = np.arange(len(esf))
    xo = np.linspace(0, len(esf)-1, len(esf)*oversample)
    esf_os = np.interp(xo, x, esf)

    # LSF
    lsf = np.diff(esf_os)
    if window_lsf:
        lsf *= np.hamming(len(lsf))

    # FFT -> MTF
    mtf = np.abs(np.fft.rfft(lsf))
    freqs = np.fft.rfftfreq(len(lsf), d=1/oversample)

    # Normierung
    fmin, fmax = norm_band
    mask = (freqs >= fmin) & (freqs <= fmax)
    norm = float(np.max(mtf[mask])) if np.any(mask) else float(np.max(mtf))
    if norm > 0:
        mtf = mtf / norm

    # Smooth MTF 
    k = int(mtf_smooth_kernel)
    if k < 3: k = 3
    if k % 2 == 0: k += 1
    mtf_smooth = np.convolve(mtf, np.ones(k)/k, mode="same")

    # MTF50: 
    i_peak = int(np.argmax(mtf_smooth))
    cross = _crossings_down(freqs, mtf_smooth, level=0.5, start_idx=i_peak)

    if len(cross) >= 2:
        i0, i1 = cross[1]
        mtf50 = _interp_cross(freqs, mtf_smooth, i0, i1, level=0.5)
        which = 2
    elif len(cross) == 1:
        i0, i1 = cross[0]
        mtf50 = _interp_cross(freqs, mtf_smooth, i0, i1, level=0.5)
        which = 1
    else:
        mtf50 = np.nan
        which = 0

    mtf_at_dict = {}
    for f in mtf_at_freqs:
        mtf_at_dict[float(f)] = _mtf_at(freqs, mtf_smooth, float(f))

    return freqs, mtf, mtf_smooth, float(mtf50), int(esf_axis), int(which), mtf_at_dict


def interactive_choose(gray_full, gray_roi, roi_offset, cx0, cy0, r0, candidates, init_idx=0, init_angle=None):
    ox, oy = roi_offset
    hR, wR = gray_roi.shape

    idx = int(np.clip(init_idx, 0, max(0, len(candidates)-1)))
    angle = float(init_angle if init_angle is not None else (candidates[idx][0] if candidates else 90.0))
    cx, cy, r = float(cx0), float(cy0), float(r0)
    roi_size = int(ROI_SIZE_DEFAULT)

    win = "MTF-ROI Interaktiv (Enter=OK, n/p Peak, a/d rotate, arrows 1px, i/j/k/l 5px, +/- ROI, s skip, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def draw():
        vis = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)

        x0,y0,w,h = SEARCH_ROI
        cv2.rectangle(vis, (x0,y0), (x0+w, y0+h), (0,255,0), 2)

        cv2.circle(vis, (int(cx+ox), int(cy+oy)), int(r), (120,120,120), 2)

        p1_roi, p2_roi = angle_to_line_points(cx, cy, angle, wR, hR)
        p1_full = (p1_roi[0]+ox, p1_roi[1]+oy)
        p2_full = (p2_roi[0]+ox, p2_roi[1]+oy)
        cv2.line(vis, p1_full, p2_full, (0,0,255), 2)

        half = roi_size//2
        mx_full, my_full = int(cx+ox), int(cy+oy)
        cv2.rectangle(vis, (mx_full-half, my_full-half), (mx_full+half, my_full-half+roi_size), (255,0,0), 2)
        cv2.circle(vis, (mx_full, my_full), 6, (0,255,255), -1)

        cand_txt = ""
        if candidates:
            cand_txt = f"Peak {idx+1}/{len(candidates)}: ang={candidates[idx][0]:.1f} val={candidates[idx][1]:.2f}"
        txt1 = f"chosen ang={angle:.1f}  center=({cx:.1f},{cy:.1f})  ROI={roi_size}px"
        cv2.putText(vis, txt1, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        if cand_txt:
            cv2.putText(vis, cand_txt, (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        crop = vis[y0:y0+h, x0:x0+w]
        if crop.size > 0:
            inset = cv2.resize(crop, (int(w*0.35), int(h*0.35)))
            vis[120:120+inset.shape[0], 40:40+inset.shape[1]] = inset

        cv2.imshow(win, vis)

    while True:
        draw()
        key = cv2.waitKeyEx(0)

        if key in (ord('q'), 27):
            cv2.destroyWindow(win)
            raise KeyboardInterrupt()

        if key == ord('s'):
            cv2.destroyWindow(win)
            return False, cx, cy, angle, roi_size

        if key in (13, 10):
            cv2.destroyWindow(win)
            return True, cx, cy, angle, roi_size

        if key == ord('n') and candidates:
            idx = (idx + 1) % len(candidates)
            angle = float(candidates[idx][0])
        if key == ord('p') and candidates:
            idx = (idx - 1) % len(candidates)
            angle = float(candidates[idx][0])

        if key == ord('a'):
            angle -= 0.5
        if key == ord('d'):
            angle += 0.5
        angle = angle % 180.0

        if key == ord('+') or key == ord('='):
            roi_size = min(900, roi_size + 10)
        if key == ord('-'):
            roi_size = max(MIN_ROI_SIZE, roi_size - 10)

        if key == ord('i'):
            cy -= 5
        if key == ord('k'):
            cy += 5
        if key == ord('j'):
            cx -= 5
        if key == ord('l'):
            cx += 5

        if key == 2424832:  # left
            cx -= 1
        if key == 2555904:  # right
            cx += 1
        if key == 2490368:  # up
            cy -= 1
        if key == 2621440:  # down
            cy += 1

        cx = float(np.clip(cx, 0, wR-1))
        cy = float(np.clip(cy, 0, hR-1))

def _freq_key(f: float) -> str:
   
    return f"{int(round(f*100)):03d}"

def process_video(video_path: Path, tag: str, freq, signal, ref_angle=None):
    frame0 = get_frame(video_path, REF_FRAME_INDEX)
    gray0  = to_gray(frame0)

    gray_roi, (ox, oy) = crop_with_roi(gray0, SEARCH_ROI)
    cx, cy, r = estimate_circle_from_roi(gray_roi)

    mag = gradient_mag(gray_roi)
    angles, energy = radial_energy_profile(mag, cx, cy, r)
    peaks = find_peaks_simple(angles, energy, PEAK_MIN_SEP_DEG)[:TOP_K_PEAKS]

    if peaks:
        init_idx = 0
        init_angle = peaks[0][0]
        if ref_angle is not None:
            peaks_sorted = sorted(peaks, key=lambda t: abs(t[0]-ref_angle))
            init_angle = peaks_sorted[0][0]
            init_idx = peaks.index(peaks_sorted[0]) if peaks_sorted[0] in peaks else 0
    else:
        init_idx = 0
        init_angle = ref_angle if ref_angle is not None else 90.0

    accepted, cx_adj, cy_adj, ang_adj, roi_size = interactive_choose(
        gray0, gray_roi, (ox, oy), cx, cy, r, peaks, init_idx=init_idx, init_angle=init_angle
    )

    base = dict(
        tag=tag,
        frequency_hz=freq if freq else "",
        signal=signal,
        video=video_path.name,
        chosen_angle_deg=np.nan,
        esf_axis=np.nan,
        mtf50_crossing=np.nan,
        mtf50_ref=np.nan,
        mtf50_mean=np.nan,
        mtf50_median=np.nan,
        mtf50_std=np.nan,
        n_frames=0,
        note=""
    )
    for f in MTF_AT_FREQS:
        k = _freq_key(f)
        base[f"mtf_at_{k}_ref"] = np.nan
        base[f"mtf_at_{k}_mean"] = np.nan
        base[f"mtf_at_{k}_median"] = np.nan
        base[f"mtf_at_{k}_std"] = np.nan

    if not accepted:
        base["note"] = "SKIPPED_BY_USER"
        return base, ref_angle

    if ref_angle is None:
        ref_angle = float(ang_adj)

    p1_roi, p2_roi = angle_to_line_points(cx_adj, cy_adj, ang_adj, gray_roi.shape[1], gray_roi.shape[0])
    p1_full = (int(p1_roi[0]+ox), int(p1_roi[1]+oy))
    p2_full = (int(p2_roi[0]+ox), int(p2_roi[1]+oy))

    mx_full = float(cx_adj + ox)
    my_full = float(cy_adj + oy)

    roi0, _ = extract_rotated_roi(gray0, p1_full, p2_full, mx_full, my_full, roi_size)
    if roi0 is None:
        base["chosen_angle_deg"] = float(ang_adj)
        base["note"] = "INVALID_ROI"
        return base, ref_angle

    freqs_m, mtf, mtf_smooth, mtf50_ref, esf_axis, which_cross, mtf_at_ref = mtf_from_roi_normalized(roi0)

    mtf50_vals = []
    mtf_at_vals = {f: [] for f in MTF_AT_FREQS}
    which_vals = []

    for frame in iter_frames(video_path):
        gray = to_gray(frame)
        roi, _ = extract_rotated_roi(gray, p1_full, p2_full, mx_full, my_full, roi_size)
        if roi is None:
            continue
        _, _, mtf_smooth_f, m50, esf_axis_f, wc, mtf_at_f = mtf_from_roi_normalized(roi)
        if not np.isnan(m50):
            mtf50_vals.append(m50)
            which_vals.append(wc)
        for f in MTF_AT_FREQS:
            v = mtf_at_f.get(float(f), np.nan)
            if not np.isnan(v):
                mtf_at_vals[f].append(v)

    mtf50_vals = np.array(mtf50_vals, dtype=float)
    if mtf50_vals.size == 0:
        base["chosen_angle_deg"] = float(ang_adj)
        base["esf_axis"] = int(esf_axis)
        base["mtf50_crossing"] = int(which_cross)
        base["mtf50_ref"] = float(mtf50_ref)
        base["note"] = "NO_VALID_FRAMES"
        return base, ref_angle

    base["chosen_angle_deg"] = float(ang_adj)
    base["esf_axis"] = int(esf_axis)
    base["mtf50_crossing"] = int(which_cross)
    base["mtf50_ref"] = float(mtf50_ref)
    base["mtf50_mean"] = float(np.mean(mtf50_vals))
    base["mtf50_median"] = float(np.median(mtf50_vals))
    base["mtf50_std"] = float(np.std(mtf50_vals))
    base["n_frames"] = int(mtf50_vals.size)

    # MTF@f metrics
    for f in MTF_AT_FREQS:
        k = _freq_key(f)
        base[f"mtf_at_{k}_ref"] = float(mtf_at_ref.get(float(f), np.nan))
        arr = np.array(mtf_at_vals[f], dtype=float)
        if arr.size > 0:
            base[f"mtf_at_{k}_mean"] = float(np.mean(arr))
            base[f"mtf_at_{k}_median"] = float(np.median(arr))
            base[f"mtf_at_{k}_std"] = float(np.std(arr))

    if SAVE_MTF_CURVE_IMAGE:
        plt.figure(figsize=(7, 4))
        plt.plot(freqs_m, mtf, linewidth=1.6, label="MTF")
        plt.plot(freqs_m, mtf_smooth, linewidth=1.2, label=f"MTF (glatt, k={MTF_SMOOTH_KERNEL})")
        plt.axhline(0.5, linestyle="--", linewidth=1.0)

    
        for f in MTF_AT_FREQS:
            v = mtf_at_ref.get(float(f), np.nan)
            if not np.isnan(v):
                plt.plot([f], [v], marker="o", markersize=5)

        plt.xlim(*PLOT_XLIM)
        plt.ylim(*PLOT_YLIM)
        plt.xlabel("Raumfrequenz [cycles/pixel]")
        plt.ylabel("MTF [-]")

        axis_txt = "axis=0 (ESF entlang x)" if esf_axis == 0 else "axis=1 (ESF entlang y)"
        cross_txt = f"{which_cross}. Down-Crossing" if which_cross in (1,2) else "no crossing"
        plt.title(f"{tag} – MTF50={mtf50_ref:.3f} cy/px ({axis_txt}, {cross_txt})")

        plt.grid(True, alpha=0.3)
        plt.legend()
        export_plot(OUT_DIR / f"MTFcurve_{tag}.png")

    if SAVE_ROIROTATED_IMAGE:
        cv2.imwrite(str(OUT_DIR / f"ROIrotated_{tag}.png"), roi0)

    return base, ref_angle

# Summary plots vs. frequency
def make_summary_plots(df: pd.DataFrame):
    
    df2 = df.copy()
    df2["frequency_hz_num"] = pd.to_numeric(df2["frequency_hz"], errors="coerce")
    df2 = df2[~df2["frequency_hz_num"].isna()].copy()
    if df2.empty:
        return

    def plot_metric(metric_col: str, title: str, out_name: str, ylabel: str):
        plt.figure(figsize=(7,4))
        for signal, style in [("Sinus", "-o"), ("Rechteck", "-s")]:
            sub = df2[df2["signal"] == signal].sort_values("frequency_hz_num")
            if sub.empty:
                continue
            plt.plot(sub["frequency_hz_num"], sub[metric_col], style, label=signal)
        plt.xlabel("Anregungsfrequenz [Hz]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        export_plot(OUT_DIR / out_name)

    # MTF50 mean/median 
    if "mtf50_mean" in df2.columns:
        plot_metric("mtf50_mean", "MTF50 (Mittelwert über Frames) vs. Anregungsfrequenz", "Summary_MTF50_mean.png", "MTF50 [cy/px]")
    if "mtf50_median" in df2.columns:
        plot_metric("mtf50_median", "MTF50 (Median über Frames) vs. Anregungsfrequenz", "Summary_MTF50_median.png", "MTF50 [cy/px]")
    if "mtf50_std" in df2.columns:
        plot_metric("mtf50_std", "MTF50 (Std über Frames) vs. Anregungsfrequenz", "Summary_MTF50_std.png", "Std(MTF50) [cy/px]")

    # Fixed-frequency MTF metrics
    for f in MTF_AT_FREQS:
        k = _freq_key(f)
        col = f"mtf_at_{k}_mean"
        if col in df2.columns:
            plot_metric(col, f"MTF@{f:.2f} cy/px (Mittelwert über Frames) vs. Anregungsfrequenz", f"Summary_MTF_at_{k}_mean.png", f"MTF@{f:.2f} [-]")
        colm = f"mtf_at_{k}_median"
        if colm in df2.columns:
            plot_metric(colm, f"MTF@{f:.2f} cy/px (Median über Frames) vs. Anregungsfrequenz", f"Summary_MTF_at_{k}_median.png", f"MTF@{f:.2f} [-]")
        cols = f"mtf_at_{k}_std"
        if cols in df2.columns:
            plot_metric(cols, f"MTF@{f:.2f} cy/px (Std über Frames) vs. Anregungsfrequenz", f"Summary_MTF_at_{k}_std.png", f"Std(MTF@{f:.2f}) [-]")


def main():
    results = []
    freq_dirs, ruhe_dir = list_freq_dirs(BASE_DIR)

    ruhe_video = find_video(["MTF_Ruhe*.mp4", "*Ruhe*.mp4"], ruhe_dir) if ruhe_dir else None
    if not ruhe_video:
        raise RuntimeError("Ruhe-Video nicht gefunden.")

    print("Starte MTF-Interaktiv (Auto-ESF-Achse, MTF50=2. crossing sonst 1., + MTF@fixen Frequenzen)")
    print(f"[INFO] Use green channel: {USE_GREEN_CHANNEL}")
    print(f"[INFO] MTF_AT_FREQS: {MTF_AT_FREQS}")

    print(f"[INFO] Ruhe: {ruhe_video}")
    row, ref_angle = process_video(ruhe_video, "Ruhe", None, "Ruhe", ref_angle=None)
    results.append(row)
    if ref_angle is None:
        raise RuntimeError("Kein Referenzwinkel gesetzt (Ruhe wurde evtl. geskippt).")
    print(f"[INFO] Referenzwinkel gesetzt auf {ref_angle:.1f}°")

    for d in freq_dirs:
        f = int(re.findall(r"\d+", d.name)[0])
        v_s = find_video([f"MTF_Sinus_{f}Hz*.mp4", f"*Sinus*{f}Hz*.mp4"], d)
        v_r = find_video([f"MTF_Rechteck_{f}Hz*.mp4", f"*Rechteck*{f}Hz*.mp4"], d)

        if v_s:
            print(f"[INFO] Sinus {f}Hz: {v_s}")
            row, ref_angle = process_video(v_s, f"Sinus_{f}Hz", f, "Sinus", ref_angle=ref_angle)
            results.append(row)
        if v_r:
            print(f"[INFO] Rechteck {f}Hz: {v_r}")
            row, ref_angle = process_video(v_r, f"Rechteck_{f}Hz", f, "Rechteck", ref_angle=ref_angle)
            results.append(row)

    df = pd.DataFrame(results)

    out_csv = OUT_DIR / "mtf50_summary_INTERACTIVE_AUTOAXIS_2ND_CROSSING_WITH_MTF_AT.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSV: {out_csv}")

    try:
        make_summary_plots(df)
        print(f"[OK] Summary plots saved in: {OUT_DIR}")
    except Exception as e:
        print(f"[WARN] Summary plot generation failed: {e}")

    print("[DONE]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Abgebrochen durch Benutzer (q/ESC).")
