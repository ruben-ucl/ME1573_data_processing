import os, sys, functools, h5py, pywt, glob, argparse
import numpy as np
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.signal import savgol_filter

print = functools.partial(print, flush=True)
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from tools import get_paths, get_cwt_scales, get_substrate_surface_coords, define_column_labels

# ─── Command Line Arguments ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Generate animated CWT visualisation')
parser.add_argument('--trackid', type=str, required=True,
                    help='Trackid to animate, e.g. 1112_01')
parser.add_argument('--format', type=str, default='mp4', choices=['mp4', 'gif'],
                    help='Output format: mp4 (default) or gif')
parser.add_argument('--fps', type=int, default=None,
                    help='Playback frame rate (default: 30 for mp4, 15 for gif)')
parser.add_argument('--mad_threshold', type=float, default=None, metavar='THRESHOLD',
                    help='MAD outlier removal threshold, e.g. 4.0 (default: off)')
parser.add_argument('--savgol_window', type=int, default=None, metavar='WINDOW',
                    help='Savitzky-Golay smoothing window length, must be odd, e.g. 11 (default: off)')
parser.add_argument('--savgol_polyorder', type=int, default=3, metavar='ORDER',
                    help='Savitzky-Golay polynomial order (default: 3)')
args = parser.parse_args()

# ─── Global Settings ───────────────────────────────────────────────────────────

folder = get_paths()['hdf5']

# group, time, series = ('KH', 'time', 'max_depth')
group, time, series = ('AMPM', 'Time', 'Photodiode1Bits')

# Trim signal
trim_start = 510 if group == 'AMPM' else 0
trim_end   = 510 if group == 'AMPM' else 0

#Trim x-ray video
xray_skip = 52 if group == 'AMPM' else 48    # skip this many frames from the front of the x-ray stack,
                                             # excess tail frames are clipped automatically

series_name = f'{group}_{series}'
cwt_cmap = 'jet'
plt.rcParams.update({'font.size': 10})

wavelet = 'cmor2.5-0.5'
cwt_scales_num = 256
pad_factor = 2
freq_min = 1000   # Hz
freq_max = 20000  # Hz

fps   = 30    # default; overridden by --fps, or lowered to 15 for gif if not specified
speed = 1.0  # >1.0 faster, <1.0 slower than real-time
crop_height = 300  # height in pixels of the x-ray crop window
xray_rate = 40000  # Hz — frame rate of bs-f40 dataset


# ─── Utility Functions ─────────────────────────────────────────────────────────

def build_signal_index(n_xray_frames, xray_rate, n_signal, signal_rate):
    """
    Nearest-neighbor mapping from x-ray frame index → signal sample index.
    Animation iterates over x-ray frames; signal/CWT follow at the matched sample.
    Works for any rate ratio; identity when rates are equal.
    """
    indices = np.round(np.arange(n_xray_frames) * (signal_rate / xray_rate)).astype(int)
    return np.clip(indices, 0, n_signal - 1)


def mad_interpolate(s, threshold):
    """Replace MAD outliers with linear interpolation from neighbouring clean samples."""
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return s
    outliers = np.abs(s - med) > threshold * mad
    if not np.any(outliers):
        return s
    clean = s.copy()
    x_all = np.arange(len(s))
    x_good = x_all[~outliers]
    clean[outliers] = np.interp(x_all[outliers], x_good, s[x_good])
    return clean


def _freq_yticks_khz(fmin_hz, fmax_hz):
    """Power-of-2 tick positions in kHz spanning fmin_hz–fmax_hz, including endpoints."""
    lo, hi = fmin_hz / 1000, fmax_hz / 1000
    p2 = [2**i for i in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)
          if lo <= 2**i <= hi]
    return sorted(set([round(lo, 4)] + p2 + [round(hi, 4)]))


def compute_crop_rows(xray_frames, trackid, folder, crop_height=300):
    """Determine crop row indices so the substrate surface appears at the top-third."""
    csv_path = Path(folder, 'substrate_surface_measurements', 'substrate_surface_locations.csv')
    try:
        frame_shape = (1, xray_frames.shape[1], xray_frames.shape[2])
        _, yy = get_substrate_surface_coords(frame_shape, csv_path, trackid)
        substrate_row = yy[xray_frames.shape[2] // 2]   # centre column
    except Exception:
        return 150, 450   # fallback matching the static script's hardcoded crop
    crop_start = max(0, substrate_row - crop_height // 3)
    crop_end = crop_start + crop_height
    return crop_start, crop_end

# ─── Main ──────────────────────────────────────────────────────────────────────

trackid = args.trackid
matches = glob.glob(f'{folder}/{trackid}*.hdf5')
if not matches:
    print(f"ERROR: No HDF5 file found for trackid '{trackid}' in {folder}")
    sys.exit(1)
filepath = sorted(matches)[0]
print(f'Loading: {filepath}')

# ─── Data Loading ──────────────────────────────────────────────────────────────

with h5py.File(filepath, 'r') as file:
    t = np.array(file[f'{group}/{time}'])[trim_start:-trim_end if trim_end else None]
    s = np.array(file[f'{group}/{series}'])[trim_start:-trim_end if trim_end else None]
    xray_frames = np.array(file['bs-f40'])

if args.mad_threshold is not None:
    _med = np.median(s)
    _abs_dev = np.abs(s - _med)
    _mad = np.median(_abs_dev)
    _pcts = np.percentile(_abs_dev, [75, 90, 95, 98, 99])
    print(f'MAD stats — signal median: {_med:.4g},  MAD: {_mad:.4g}')
    def _fmt(v): return f'{v:.4g} (x{v/_mad:.2f})'
    print(f'  Abs. deviation percentiles:  '
          f'p75={_fmt(_pcts[0])}  p90={_fmt(_pcts[1])}  p95={_fmt(_pcts[2])}  '
          f'p98={_fmt(_pcts[3])}  p99={_fmt(_pcts[4])}  max={_fmt(_abs_dev.max())}')
    print(f'  Threshold {args.mad_threshold} × MAD = {args.mad_threshold * _mad:.4g}  '
          f'({(_abs_dev > args.mad_threshold * _mad).sum()} sample(s) flagged)')
    s = mad_interpolate(s, args.mad_threshold)
    print(f'MAD outlier removal applied')
if args.savgol_window is not None:
    s = savgol_filter(s, window_length=args.savgol_window, polyorder=args.savgol_polyorder)
    print(f'Savgol filter applied (window={args.savgol_window}, polyorder={args.savgol_polyorder})')

# Compute sampling_period early so it can be used in alignment below
sampling_period = round(t[1] - t[0], 9)

# Align x-ray stack with signal: skip leading blank frames
if xray_skip > 0:
    xray_frames = xray_frames[xray_skip:]

# Build nearest-neighbor signal index mapping: one entry per x-ray frame
signal_indices = build_signal_index(len(xray_frames), xray_rate, len(s), round(1 / sampling_period))

print(f'X-ray frames: {len(xray_frames)} @ {xray_rate/1000:.0f} kHz, '
      f'Signal length: {len(s)} @ {round(1/sampling_period)/1000:.0f} kHz '
      f'(ratio {round(1/sampling_period)/xray_rate:.2f}x)')
if signal_indices[-1] >= len(s):
    raise ValueError(f"Signal too short for x-ray stack after skipping {xray_skip} frames.")

# Contrast enhancement: clip to [2nd, 98th] percentile across the whole stack,
# then normalise to [0, 1] so imshow renders consistently across all frames.
p_lo, p_hi = np.percentile(xray_frames, (2, 98))
xray_frames = np.clip(xray_frames, p_lo, p_hi).astype(np.float32)
xray_frames = (xray_frames - p_lo) / (p_hi - p_lo)
print(f'Contrast normalised: p2={p_lo:.1f}, p98={p_hi:.1f}')

# ─── CWT Computation ───────────────────────────────────────────────────────────

sampling_duration = round(t[-1] - t[0], 9)
sampling_rate    = round(1 / sampling_period, 7)

nyquist_freq = sampling_rate / 2
min_freq     = 1 / sampling_duration

fmin = min_freq if freq_min == 'auto' else freq_min
fmax = nyquist_freq if freq_max == 'auto' else freq_max
print(f'Frequency range: {fmin/1000:.4g}–{fmax/1000:.4g} kHz')

scales, _ = get_cwt_scales(wavelet, num=cwt_scales_num, sampling_rate=sampling_rate,
                            fmin=fmin, fmax=fmax)

pad_width = pad_factor * len(s)
s_pad = np.pad(s, pad_width, mode='symmetric')
print('Computing CWT...')
cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period)
freqs = freqs.real
cwtmatr = np.abs(cwtmatr[:, pad_width:pad_width + len(s)])
cwtmatr = cwtmatr / cwtmatr.max()   # normalise to [0, 1]
print('CWT done.')

t_ax = t * 1000          # ms, 1-D for axvline / dot
f_ax_2d, t_ax_2d = np.meshgrid(freqs / 1000, t * 1000, indexing='ij')

# ─── Crop Row Computation ──────────────────────────────────────────────────────

crop_start, crop_end = compute_crop_rows(xray_frames, trackid, folder, crop_height)
print(f'X-ray crop rows: {crop_start}–{crop_end}')

# ─── Figure Setup — static elements ───────────────────────────────────────────

series_label = define_column_labels().get(series, [None, series])[1]

fig = plt.figure(figsize=[7, 7], dpi=150)
# Outer: x-ray on top, signal+CWT group below — gap between sections controlled
# independently from the gap between signal and CWT.
outer = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 2],
                 hspace=0.04, top=0.93, bottom=0.08, left=0.13, right=0.88)
gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                  width_ratios=[97, 3], wspace=0.02)
gs_bot = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1],
                                  height_ratios=[1, 1.2], width_ratios=[97, 3],
                                  hspace=0.06, wspace=0.02)

ax_xray  = fig.add_subplot(gs_top[0, 0])
ax_xrayb = fig.add_subplot(gs_top[0, 1])
ax_sig   = fig.add_subplot(gs_bot[0, 0])
ax_sigb  = fig.add_subplot(gs_bot[0, 1])
ax_cwt   = fig.add_subplot(gs_bot[1, 0], sharex=ax_sig)
ax_cbar  = fig.add_subplot(gs_bot[1, 1])

fig.suptitle(trackid, fontsize=10)

# Panel 1 (top): x-ray, animated
im = ax_xray.imshow(xray_frames[0][crop_start:crop_end], cmap='gray', animated=True,
                    aspect='equal')
scalebar = ScaleBar(4.3, 'µm', length_fraction=0.15, width_fraction=0.02,
                    frameon=False, color='w', location='lower right',
                    rotation='horizontal-only')
ax_xray.add_artist(scalebar)
ax_xray.set_anchor('S')   # letterbox whitespace goes above, not between image and signal
ax_xray.axis('off')
ax_xrayb.axis('off')

# Panel 2 (middle): signal
ax_sig.plot(t * 1000, s, lw=0.75, color='steelblue')
ax_sig.set_xlim(t[0] * 1000, t[-1] * 1000)
ax_sig.set_ylabel(series_label)
if series == 'area':
    ax_sig.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.setp(ax_sig.get_xticklabels(), visible=False)  # x-label shown on CWT below
ax_sigb.axis('off')

# Panel 3 (bottom): CWT spectrogram
pcm = ax_cwt.pcolormesh(t_ax_2d, f_ax_2d, cwtmatr, cmap=cwt_cmap, vmin=0, vmax=1)
ax_cwt.set_yscale('log', base=2)
ax_cwt.set_ylim(fmin / 1000, fmax / 1000)
ax_cwt.set_yticks(_freq_yticks_khz(fmin, fmax))
ax_cwt.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
ax_cwt.set_xlabel('Time [ms]')
ax_cwt.set_ylabel('Frequency [kHz]')
fig.colorbar(pcm, cax=ax_cbar, label='Intensity')

# ─── Animated Artists ─────────────────────────────────────────────────────────

dot,  = ax_sig.plot([t[0] * 1000], [s[0]], 'o', color='red', ms=5, zorder=5)
vline = ax_cwt.axvline(t[0] * 1000, color='white', lw=0.8, alpha=0.85)

# ─── FuncAnimation ────────────────────────────────────────────────────────────

def update(i):
    si = signal_indices[i]
    dot.set_data([t[si] * 1000], [s[si]])
    vline.set_xdata([t[si] * 1000])
    im.set_data(xray_frames[i][crop_start:crop_end])
    return dot, vline, im

interval_ms = (1 / xray_rate * 1000) / speed
anim = FuncAnimation(fig, update, frames=len(xray_frames), interval=interval_ms, blit=True)

# ─── Save ──────────────────────────────────────────────────────────────────────

def _make_writer(fmt, fps):
    if fmt == 'gif':
        from matplotlib.animation import PillowWriter
        return PillowWriter(fps=fps)
    return FFMpegWriter(fps=fps)

fmt = args.format
if args.fps is not None:
    fps = args.fps
elif fmt == 'gif':
    fps = 15   # sensible default for gif; mp4 keeps the module-level 30

output_folder = Path(folder, 'CWT', series_name, wavelet, cwt_cmap, 'animations')
os.makedirs(output_folder, exist_ok=True)
out_path = Path(output_folder, f'{trackid}_{series}_CWT_animated.{fmt}')

print(f'Saving {fmt.upper()} ({len(xray_frames)} frames @ {fps} fps)...')
anim.save(str(out_path), writer=_make_writer(fmt, fps))
print(f'Saved: {out_path}')
