import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from scipy.signal import spectrogram


def plot_spectrogram(signal, sampling_rate, nperseg=128, noverlap=64, cmap='plasma', title='Spectrogram', plot='sqrt'):
    f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

    Sxx_sqrt = np.sqrt(Sxx)
    Sxx_softlog = np.log1p(Sxx)  # ≈ log(1 + Sxx)
    gamma = 0.01  # between 0 (strong compression) and 1 (no compression)
    Sxx_gamma = Sxx ** gamma  # gamma = 0.5 → square root || gamma = 0.3 → softer than sqrt but not as compressed as log
    Sxx_asinh = np.arcsinh(Sxx)
    Sxx_log2 = np.log2(Sxx + 1e-10)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # avoid log(0) POwer spectrum in  DB

    if plot == 'sqrt':
        plt.figure(figsize=(20, 4))
        ##plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_sqrt, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power sqrt')
        plt.tight_layout()
        plt.show()

    if plot == 'softlog':
        plt.figure(figsize=(20, 4))
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_softlog, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power Sxx_softlog')
        plt.tight_layout()
        plt.show()

    if plot == 'gamma':
        plt.figure(figsize=(20, 4))
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_gamma, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power Sxx_gamma')
        plt.tight_layout()
        plt.show()

    if plot == 'asinh':
        plt.figure(figsize=(20, 4))
        ##plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_asinh, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power Sxx_asinh')
        plt.tight_layout()
        plt.show()

    if plot == 'Log2':
        plt.figure(figsize=(20, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_log2, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power Log2')
        plt.tight_layout()
        plt.show()

    if plot == 'dB':
        plt.figure(figsize=(20, 4))
        # plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap=cmap)
        plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap=cmap)
        plt.title(title)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()
        plt.show()


def func_fourier_with_power_plot(df, col, frequency_threshold=0.1, sampling_rate_hz=1.0, plot=True, plot_power=False,
                                 plot_phase=False, plot_PowerLaw=False):
    """
    Applies FFT-based low-pass filtering, computes power & phase spectra, and displays diagnostic plots.

    Parameters:
        df (pd.DataFrame): Time-series DataFrame.
        col (str): Target column.
        frequency_threshold (float): Low-pass cutoff frequency in Hz.
        sampling_rate_hz (float): Sampling rate in Hz (e.g. 1/86400 for daily).
        plot (bool): Toggle for plotting results.

    Returns:
        pd.DataFrame with filtered signal and spectral metadata.
    """

    df = df.copy()
    df['diff'] = df[col].diff().bfill()

    sampling_interval = 1.0 / sampling_rate_hz

    # FFT
    signal = df[col].values
    signal_diff = df['diff'].values

    frequencies_hz = np.fft.fftfreq(len(signal), d=sampling_interval)
    frequencies_hz_diff = np.fft.fftfreq(len(signal_diff), d=sampling_interval)

    fft_values = fft(signal)
    fft_values_diff = fft(signal_diff)

    power_spectrum = np.abs(fft_values) ** 2
    power_spectrum_diff = np.abs(fft_values_diff) ** 2

    phase_spectrum = np.angle(fft_values)
    phase_spectrum_diff = np.angle(fft_values_diff)

    # Low-pass filtering
    filtered_fft = fft_values.copy()
    filtered_fft_diff = fft_values_diff.copy()

    filtered_fft[np.abs(frequencies_hz) > frequency_threshold] = 0
    filtered_fft_diff[np.abs(frequencies_hz_diff) > frequency_threshold] = 0

    filtered_signal = ifft(filtered_fft).real
    filtered_signal_diff = ifft(filtered_fft_diff).real

    # Store in DataFrame
    df['filtered_fft'] = filtered_signal
    df['frequencies_hz'] = frequencies_hz
    df['power_spectrum'] = power_spectrum
    df['phase_spectrum'] = phase_spectrum

    df['filtered_fft_diff'] = filtered_signal_diff
    df['frequencies_hz_diff'] = frequencies_hz_diff
    df['power_spectrum_diff'] = power_spectrum_diff
    df['phase_spectrum_diff'] = phase_spectrum_diff

    if plot_power:
        mask = frequencies_hz > 0

        # Power & Phase Spectrum
        fig, axs = plt.subplots(2, 1, figsize=(20, 6))

        axs[0].stem(frequencies_hz[mask], power_spectrum[mask], basefmt=" ", label='Power')
        axs[0].set_title('Power Spectrum')
        axs[0].set_xlabel('Frequency [Hz]')
        axs[0].set_ylabel('Power')
        axs[0].grid(True)

        axs[1].plot(frequencies_hz[mask], phase_spectrum[mask])
        axs[1].set_title("Phase Spectrum")
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("Phase [radians]")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    if plot:
        # Spectrograms
        print("\n🎛️ Spectrograms:")

        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=512, noverlap=384, title='Original Signal')  #(75% overlap)
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=256, noverlap=128, title='Original Signal')  #(50% overlap)
        plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal 128',
                         plot='softlog')
        plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64,
                         title=f'{col} Signal diff 128', plot='softlog')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal 128', plot='asinh')
        # plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal diff 128', plot='asinh')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal 128', plot='gamma')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal 128', plot='Log2')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=128, noverlap=64, title=f'{col} Signal 128', plot='dB')



        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36, title=f'{col} Signal 64', plot='softlog')
        plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36, title=f'{col} Signal 64',
                         plot='asinh')
        plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36,
                         title=f'{col} Signal diff 64', plot='softlog')
        # plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=64,  noverlap=36, title=f'{col} Signal diff 64', plot='asinh')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36, title=f'{col} Signal 64', plot='gamma')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36, title=f'{col} Signal 64', plot='Log2')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=64, noverlap=36, title=f'{col} Signal 64', plot='dB')



        plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18, title=f'{col} Signal 32',
                         plot='softlog')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18, title=f'{col} Signal 32', plot='asinh')
        # plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=32,  noverlap=18, title=f'{col} Signal diff 32', plot='softlog')
        plot_spectrogram(signal_diff, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18,
                         title=f'{col} Signal diff 32', plot='asinh')

        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18, title=f'{col} Signal 32', plot='gamma')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18, title=f'{col} Signal 32', plot='Log2')
        # plot_spectrogram(signal, sampling_rate=sampling_rate_hz, nperseg=32, noverlap=18, title=f'{col} Signal 32', plot='dB')

    # plt.figure(figsize=(12.3, 3))
    # plt.plot(df.index, df[col], label=col)
    # plt.xlabel('Time')
    # plt.title(f'{col} vs Amplitude of the signal')
    # plt.legend()
    # plt.grid(True)
    # plt.margins(x=0)                  # Remove x-axis margin (optional)
    # plt.tight_layout()
    # plt.show()

    if plot_PowerLaw:
        # Power Law Slope Diagnostic
        print("\n📐 Power Law Slope (log-log):")
        log_freq = np.log10(frequencies_hz[mask])
        log_power = np.log10(power_spectrum[mask])
        slope, intercept = np.polyfit(log_freq, log_power, 1)

        plt.figure(figsize=(10, 4))
        plt.plot(log_freq, log_power, '.', label='Spectral Content')
        plt.plot(log_freq, slope * log_freq + intercept, 'r-', label=f'slope ≈ {slope:.2f}')
        plt.xlabel('log(Frequency)')
        plt.ylabel('log(Power)')
        plt.title('Log-Log Power Spectrum')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"🔎 Power-law slope ≈ {slope:.2f}")
        print("• Near -1 → pink noise | Near -2 → Brownian motion")

    return df
