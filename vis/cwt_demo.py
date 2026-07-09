import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import interp1d

np.random.seed(42)

# --- Synthetic signal ---
fs = 1000.0
duration = 4.0
t = np.arange(0, duration, 1.0 / fs)
N = len(t)

def smooth_envelope(N, n_knots=10, seed=0):
    rng = np.random.default_rng(seed)
    knots = np.linspace(0, N - 1, n_knots)
    vals = rng.uniform(0.2, 1.0, n_knots)
    f = interp1d(knots, vals, kind='cubic', fill_value='extrapolate')
    return np.clip(f(np.arange(N)), 0.05, 1.0)

component_freqs = [4, 12, 35, 75]  # Hz
signal = sum(
    smooth_envelope(N, seed=i) * np.sin(2 * np.pi * freq * t)
    for i, freq in enumerate(component_freqs)
)
signal += 6.0 + np.random.normal(0, 0.15, N)   # DC offset keeps signal positive
log_signal = np.log(signal)

# --- CWT (complex Morlet) ---
wavelet = 'cmor1.5-1.0'
dt = 1.0 / fs
cf = pywt.central_frequency(wavelet)
freqs_cwt = np.geomspace(1.0, 120.0, 150)
scales = cf / (freqs_cwt * dt)
coef_sig, freqs_out = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
coef_log, _ = pywt.cwt(log_signal, scales, wavelet, sampling_period=dt)
power_sig = np.abs(coef_sig)
power_log = np.abs(coef_log)

# --- Figure ---
fig = plt.figure(figsize=(6.30, 4.2))
gs = fig.add_gridspec(2, 2, hspace=0.04, wspace=0.50, height_ratios=[1, 1])

ax_sig = fig.add_subplot(gs[0, 0])
ax_log = fig.add_subplot(gs[1, 0], sharex=ax_sig)
ax_cwt_sig = fig.add_subplot(gs[0, 1])
ax_cwt_log = fig.add_subplot(gs[1, 1], sharex=ax_cwt_sig)

# Time-domain signal
ax_sig.plot(t, signal, lw=0.5, c='#2c7bb6')
ax_sig.set_ylabel('Amplitude', fontsize=9)
ax_sig.tick_params(labelsize=8)
plt.setp(ax_sig.get_xticklabels(), visible=False)

# Log amplitude
ax_log.plot(t, log_signal, lw=0.5, c='#d7191c')
ax_log.set_ylabel('log(Amplitude)', fontsize=9)
ax_log.set_xlabel('Time (s)', fontsize=9)
ax_log.tick_params(labelsize=8)

# CWT of signal
vmax_sig = np.percentile(power_sig, 99)
im_sig = ax_cwt_sig.pcolormesh(t, freqs_out, power_sig, cmap='plasma',
                                shading='auto', vmin=0, vmax=vmax_sig)
ax_cwt_sig.set_yscale('log')
ax_cwt_sig.set_ylabel('Frequency (Hz)', fontsize=9)
ax_cwt_sig.tick_params(labelsize=8)
plt.setp(ax_cwt_sig.get_xticklabels(), visible=False)
cb_sig = plt.colorbar(im_sig, ax=ax_cwt_sig, pad=0.03)
cb_sig.set_label('|CWT|', fontsize=8)
cb_sig.ax.tick_params(labelsize=8)

# CWT of log signal
vmax_log = np.percentile(power_log, 99)
im_log = ax_cwt_log.pcolormesh(t, freqs_out, power_log, cmap='plasma',
                                shading='auto', vmin=0, vmax=vmax_log)
ax_cwt_log.set_yscale('log')
ax_cwt_log.set_ylabel('Frequency (Hz)', fontsize=9)
ax_cwt_log.set_xlabel('Time (s)', fontsize=9)
ax_cwt_log.tick_params(labelsize=8)
cb_log = plt.colorbar(im_log, ax=ax_cwt_log, pad=0.03)
cb_log.set_label('|CWT|', fontsize=8)
cb_log.ax.tick_params(labelsize=8)

for ax in [ax_sig, ax_log, ax_cwt_sig, ax_cwt_log]:
    for sp in ax.spines.values():
        sp.set_edgecolor('black')

fig.savefig('vis/cwt_demo.pdf', bbox_inches='tight')
fig.savefig('vis/cwt_demo.png', dpi=300, bbox_inches='tight')
plt.show()
