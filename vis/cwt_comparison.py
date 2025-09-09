#!/usr/bin/env python3
"""
Compare CWT implementations using pywt, scipy, and ssqueezepy.

Generates a test signal with two different frequency sin waves and compares
the frequency scalograms produced by each package using proper scale-frequency conversion.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Import CWT packages
import pywt
from scipy import signal as scipy_signal
try:
    import ssqueezepy as ssq
    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False
    print("Warning: ssqueezepy not available, skipping ssqueezepy comparison")

def generate_test_signal(duration=2.0, fs=500):
    """Generate test signal: 5 Hz → 20 Hz with 50% amplitude modulation in middle 50%."""
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # First half: 5 Hz sine wave
    # Second half: 20 Hz sine wave
    mid_point = len(t) // 2
    
    signal = np.zeros_like(t)
    signal[:mid_point] = np.sin(2 * np.pi * 5 * t[:mid_point])
    signal[mid_point:] = np.sin(2 * np.pi * 20 * t[mid_point:])
    
    # Apply 50% amplitude modulation to middle 50% of the signal
    # Middle 50% spans from 25% to 75% of the total duration
    total_samples = len(t)
    start_idx = int(0.25 * total_samples)  # 25% point
    end_idx = int(0.75 * total_samples)    # 75% point
    
    # Create smooth window function using cosine taper
    window_length = end_idx - start_idx
    # Create a sharp transition using raised cosine window
    taper_length = int(0.1 * window_length)  # 10% of modulation region for smooth edges
    
    # Initialize modulation window (1.0 everywhere, will be modified in middle)
    modulation_window = np.ones_like(signal)
    
    # Create the modulation region (50% amplitude = 0.5 multiplier)
    modulation_window[start_idx:end_idx] = 0.5
    
    # Smooth the transitions using cosine tapers
    if taper_length > 0:
        # Left taper: transition from 1.0 to 0.5
        left_taper_start = start_idx
        left_taper_end = start_idx + taper_length
        taper_samples = np.arange(taper_length)
        left_taper = 1.0 - 0.5 * (1 - np.cos(np.pi * taper_samples / taper_length)) / 2
        modulation_window[left_taper_start:left_taper_end] = left_taper
        
        # Right taper: transition from 0.5 to 1.0
        right_taper_start = end_idx - taper_length
        right_taper_end = end_idx
        right_taper = 0.5 + 0.5 * (1 - np.cos(np.pi * taper_samples / taper_length)) / 2
        modulation_window[right_taper_start:right_taper_end] = right_taper
    
    # Apply modulation
    signal_modulated = signal * modulation_window
    
    return t, signal_modulated

def cwt_pywt(signal, fs=500, f_min=1, f_max=50, n_scales=60, wavelet='cmor1.5-1.0'):
    """Compute CWT using PyWavelets with proper scale-frequency conversion."""
    dt = 1.0 / fs
    
    # Generate frequency range and convert to scales using PyWavelets functions
    frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_scales)
    scales = []
    actual_frequencies = []
    
    for freq in frequencies:
        # Convert Hz to normalized frequency for PyWavelets
        normalized_freq = freq * dt
        scale = pywt.frequency2scale(wavelet, normalized_freq, precision=10)
        scales.append(scale)
        # Verify the actual frequency (will be in normalized units)
        actual_freq_norm = pywt.scale2frequency(wavelet, scale, precision=10)
        actual_freq = actual_freq_norm / dt  # Convert back to Hz
        actual_frequencies.append(actual_freq)
    
    scales = np.array(scales)
    actual_frequencies = np.array(actual_frequencies)
    
    # Compute CWT
    coefficients, returned_frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
    
    return coefficients, returned_frequencies

def cwt_scipy(signal, fs=500, f_min=1, f_max=50, n_scales=60, wavelet_func=scipy_signal.morlet2):
    """Compute CWT using SciPy with proper width-frequency conversion."""
    dt = 1.0 / fs
    center_freq = 1.0  # For morlet2 wavelet (Mexican hat uses different approach)
    
    # Generate frequency range and convert to widths
    frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_scales)
    widths = []
    actual_frequencies = []
    
    for freq in frequencies:
        if wavelet_func == scipy_signal.ricker:
            # Mexican hat (Ricker) wavelet - width is directly related to scale
            width = fs / freq  # Direct relationship for Mexican hat
            actual_freq = fs / width
        else:
            # Morlet2 wavelet
            width = center_freq / (freq * dt)
            actual_freq = center_freq / (width * dt)
        
        widths.append(width)
        actual_frequencies.append(actual_freq)
    
    widths = np.array(widths)
    actual_frequencies = np.array(actual_frequencies)
    
    # Compute CWT
    if wavelet_func == scipy_signal.ricker:
        coefficients = scipy_signal.cwt(signal, wavelet_func, widths)
    else:
        coefficients = scipy_signal.cwt(signal, wavelet_func, widths, w=6.0)
    
    return coefficients, actual_frequencies

def cwt_ssqueezepy(signal, fs=500, f_min=1, f_max=50, n_scales=60, wavelet='morlet'):
    """Compute CWT using ssqueezepy with proper scale-frequency conversion."""
    if not SSQUEEZEPY_AVAILABLE:
        return None, None
    
    try:
        # Get ssqueezepy's default scales first to understand the valid range
        _, default_scales = ssq.cwt(signal, wavelet, fs=fs)
        
        # Convert default scales to frequencies to find valid range
        try:
            from ssqueezepy.experimental import scale_to_freq
            default_frequencies = scale_to_freq(default_scales, wavelet, len(signal), fs=fs)
        except ImportError:
            # Fallback manual conversion - use relationship found from analysis
            dt = 1.0 / fs
            default_frequencies = 1.0 / (default_scales * dt)
        
        # Find scales that correspond to our target frequency range
        freq_mask = (default_frequencies >= f_min) & (default_frequencies <= f_max)
        if np.any(freq_mask):
            valid_scales = default_scales[freq_mask]
            valid_frequencies = default_frequencies[freq_mask]
            
            # Create logarithmic spacing within the valid scale range
            # This ensures ssqueezepy recognizes it as exponential spacing
            scale_min = valid_scales.min()
            scale_max = valid_scales.max()
            
            # Generate logarithmically spaced scales within valid range
            target_scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
            
            # Use ssqueezepy with these properly spaced scales
            coefficients, scales_out = ssq.cwt(signal, wavelet, scales=target_scales, fs=fs)
            
            # Get the actual frequencies achieved
            try:
                from ssqueezepy.experimental import scale_to_freq
                actual_frequencies = scale_to_freq(scales_out, wavelet, len(signal), fs=fs)
            except ImportError:
                # Fallback manual conversion
                dt = 1.0 / fs
                actual_frequencies = 1.0 / (scales_out * dt)
            
            return coefficients, actual_frequencies
        else:
            print("ssqueezepy: No scales found in target frequency range")
            return None, None
            
    except Exception as e:
        print(f"ssqueezepy CWT failed: {e}")
        return None, None


def plot_comparison(t, signal, results, save_path=None):
    """Plot original signal and CWT results for comparison."""
    from matplotlib.gridspec import GridSpec
    
    # Create figure with GridSpec for better control over colorbar placement
    fig = plt.figure(figsize=(12, 20))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1, 0.1], hspace=0.5)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    cbar_ax = fig.add_subplot(gs[4])
    
    # Common colormap
    cmap = 'viridis'
    
    # Plot original signal
    axes[0].plot(t, signal, 'b-', linewidth=2)
    axes[0].set_title('Original Signal: 5 Hz → 20 Hz (50% amplitude modulation in middle 50%)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_xlim(0, 2)
    axes[0].grid(True, alpha=0.3)
    
    # Use independent min-max normalization for each CWT result
    # This will make each colorbar range from 0 to 1
    
    # Plot CWT results
    plot_idx = 1
    last_im = None  # Keep track of the last image for colorbar
    
    for name, (coeffs, freqs) in results.items():
        if coeffs is None:
            continue
            
        ax = axes[plot_idx]
        
        # Always display with frequencies in ascending order (low to high)
        if freqs[0] > freqs[-1]:
            # Frequencies are in descending order, reverse both coeffs and freqs
            coeffs_display = np.abs(coeffs)[::-1, :]
            freqs_display = freqs[::-1]
        else:
            coeffs_display = np.abs(coeffs)
            freqs_display = freqs
        
        # Apply min-max normalization to this specific CWT result
        coeffs_min = np.min(coeffs_display)
        coeffs_max = np.max(coeffs_display)
        coeffs_normalized = (coeffs_display - coeffs_min) / (coeffs_max - coeffs_min)
        
        # Use pcolormesh for proper frequency mapping instead of imshow
        # pcolormesh correctly handles non-uniform frequency spacing
        T, F = np.meshgrid(t, freqs_display)
        im = ax.pcolormesh(T, F, coeffs_normalized, 
                          cmap=cmap, shading='auto',
                          vmin=0, vmax=1)
        
        # Store the last image for colorbar
        last_im = im
        
        ax.set_title(f'CWT Scalogram - {name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, 2)
        
        # Set y-axis to log base 2 scale
        ax.set_yscale('log', base=2)
        
        # Set specific y-axis ticks at 2, 4, 8, 16, 32 Hz
        desired_ticks = [2, 4, 8, 16, 32]
        f_min, f_max = freqs_display[0], freqs_display[-1]
        
        # Only show ticks that are within the frequency range
        valid_ticks = [tick for tick in desired_ticks if tick >= f_min and tick <= f_max]
        
        ax.set_yticks(valid_ticks)
        ax.set_yticklabels([str(tick) for tick in valid_ticks])
        
        # Add horizontal reference lines at expected frequencies
        # Find the closest frequencies to 5 Hz and 20 Hz in the display array
        freq_5hz = freqs_display[np.argmin(np.abs(freqs_display - 5))]
        freq_20hz = freqs_display[np.argmin(np.abs(freqs_display - 20))]
        
        ax.axhline(y=freq_5hz, color='white', linestyle=':', alpha=0.8, linewidth=1.5, 
                  label=f'5 Hz ({freq_5hz:.1f})')
        ax.axhline(y=freq_20hz, color='white', linestyle='--', alpha=0.8, linewidth=1.5, 
                  label=f'20 Hz ({freq_20hz:.1f})')
        ax.legend(loc='lower right')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    # Add horizontal colorbar at the bottom
    if last_im is not None:
        plt.colorbar(last_im, cax=cbar_ax, orientation='horizontal', 
                    label='Normalized |CWT Coefficient|')
    else:
        cbar_ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    return fig

def run_cwt_comparison(t, test_signal, fs, f_min, f_max, n_scales, wavelet_config, output_suffix):
    """Run CWT comparison with specified wavelet configuration."""
    pywt_wavelet, scipy_wavelet, ssq_wavelet = wavelet_config
    
    print(f"\n{'='*60}")
    print(f"CWT Comparison - {output_suffix}")
    print(f"{'='*60}")
    print(f"PyWavelets: {pywt_wavelet}")
    print(f"SciPy: {scipy_wavelet.__name__ if hasattr(scipy_wavelet, '__name__') else str(scipy_wavelet)}")
    print(f"ssqueezepy: {ssq_wavelet}")
    
    # Store results
    results = {}
    
    # PyWavelets CWT
    print(f"\nComputing CWT with PyWavelets ({pywt_wavelet})...")
    try:
        coeffs_pywt, freqs_pywt = cwt_pywt(test_signal, fs=fs, f_min=f_min, f_max=f_max, 
                                           n_scales=n_scales, wavelet=pywt_wavelet)
        results[f'PyWavelets ({pywt_wavelet})'] = (coeffs_pywt, freqs_pywt)
        print("✓ PyWavelets CWT completed")
    except Exception as e:
        print(f"✗ PyWavelets CWT failed: {e}")
        results[f'PyWavelets ({pywt_wavelet})'] = (None, None)
    
    # SciPy CWT
    scipy_name = scipy_wavelet.__name__ if hasattr(scipy_wavelet, '__name__') else str(scipy_wavelet)
    print(f"Computing CWT with SciPy ({scipy_name})...")
    try:
        coeffs_scipy, freqs_scipy = cwt_scipy(test_signal, fs=fs, f_min=f_min, f_max=f_max, 
                                              n_scales=n_scales, wavelet_func=scipy_wavelet)
        results[f'SciPy ({scipy_name})'] = (coeffs_scipy, freqs_scipy)
        print("✓ SciPy CWT completed")
    except Exception as e:
        print(f"✗ SciPy CWT failed: {e}")
        results[f'SciPy ({scipy_name})'] = (None, None)
    
    # ssqueezepy CWT
    if SSQUEEZEPY_AVAILABLE:
        print(f"Computing CWT with ssqueezepy ({ssq_wavelet})...")
        try:
            coeffs_ssq, freqs_ssq = cwt_ssqueezepy(test_signal, fs=fs, f_min=f_min, f_max=f_max, 
                                                   n_scales=n_scales, wavelet=ssq_wavelet)
            if coeffs_ssq is not None:
                results[f'ssqueezepy ({ssq_wavelet})'] = (coeffs_ssq, freqs_ssq)
                print("✓ ssqueezepy CWT completed")
            else:
                print("✗ ssqueezepy CWT returned None")
                results[f'ssqueezepy ({ssq_wavelet})'] = (None, None)
        except Exception as e:
            print(f"✗ ssqueezepy CWT failed: {e}")
            results[f'ssqueezepy ({ssq_wavelet})'] = (None, None)
    
    # Plot comparison
    print(f"\nGenerating comparison plot...")
    save_path = f"D:/ME1573_data_processing/cwt_comparison_{output_suffix.lower().replace(' ', '_')}.png"
    fig = plot_comparison(t, test_signal, results, save_path=save_path)
    
    # Print summary
    print(f"\nComparison completed!")
    successful_methods = [name for name, (coeffs, _) in results.items() if coeffs is not None]
    print(f"Successful methods: {len(successful_methods)}")
    for method in successful_methods:
        coeffs, freqs = results[method]
        if coeffs is not None:
            print(f"  - {method}: {coeffs.shape[0]} scales, {freqs.min():.1f}-{freqs.max():.1f} Hz")
    
    return results

def main():
    """Main function to run CWT comparison."""
    print("CWT Package Comparison")
    print("="*50)
    
    # Centralized CWT parameters for consistent comparison
    fs = 500           # Sampling frequency
    f_min = 1          # Minimum frequency (Hz)
    f_max = 50         # Maximum frequency (Hz) 
    n_scales = 128      # Number of frequency scales
    
    # Generate test signal
    print("Generating test signal...")
    t, test_signal = generate_test_signal(duration=2.0, fs=fs)
    print(f"Signal: 5 Hz (0-1s) → 20 Hz (1-2s) with 50% amplitude modulation (0.5-1.5s), sampling at {fs} Hz")
    
    print(f"\nCWT Parameters:")
    print(f"  Frequency range: {f_min}-{f_max} Hz")
    print(f"  Number of scales: {n_scales}")
    print(f"  Sampling frequency: {fs} Hz")
    
    # Define wavelet configurations
    # Configuration 1: Morlet-type wavelets (original)
    morlet_config = (
        'cmor1.5-1.0',           # PyWavelets: Complex Morlet
        scipy_signal.morlet2,     # SciPy: Morlet2
        'morlet'                 # ssqueezepy: Morlet
    )
    
    # Configuration 2: Mexican Hat wavelets
    mexican_hat_config = (
        'mexh',                  # PyWavelets: Mexican Hat
        scipy_signal.ricker,     # SciPy: Ricker (Mexican Hat)
        'gmw'                    # ssqueezepy: Generalized Morse wavelet (closest to Mexican Hat)
    )
    
    # Run comparison with Morlet-type wavelets
    print("\n" + "="*80)
    print("COMPARISON 1: MORLET-TYPE WAVELETS")
    print("="*80)
    morlet_results = run_cwt_comparison(t, test_signal, fs, f_min, f_max, n_scales, 
                                        morlet_config, "Morlet")
    
    # Run comparison with Mexican Hat wavelets  
    print("\n" + "="*80)
    print("COMPARISON 2: MEXICAN HAT WAVELETS")
    print("="*80)
    mexican_hat_results = run_cwt_comparison(t, test_signal, fs, f_min, f_max, n_scales, 
                                             mexican_hat_config, "Mexican_Hat")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Signal: 5 Hz (0-1s) → 20 Hz (1-2s) with 50% amplitude modulation (0.5-1.5s)")
    print(f"CWT Parameters: {f_min}-{f_max} Hz, {n_scales} scales, {fs} Hz sampling")
    print(f"Output files generated:")
    print(f"  - cwt_comparison_morlet.png (Morlet-type wavelets)")
    print(f"  - cwt_comparison_mexican_hat.png (Mexican Hat wavelets)")
    
    return morlet_results, mexican_hat_results

if __name__ == "__main__":
    main()