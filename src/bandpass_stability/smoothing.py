"""
De-noise CASA bandpass calibration tables with a median filter along
the frequency axis.

Smoothing is performed on log-amplitude and unwrapped phase
independently, inside each contiguous run of unflagged channels.
Flagged channels are never smoothed across and are left unchanged.
"""

import glob
import os
import shutil

import numpy as np
from scipy.ndimage import median_filter

from .caltable_utils import iter_pol_spectra, natural_sort_key, set_pol_spectrum


# ============================================================
# File helpers
# ============================================================

def find_input_tables(pattern, output_tag):
    """
    Find and naturally sort input bandpass tables, excluding any
    table that already carries `output_tag` (i.e. outputs from a
    previous run of this same step).
    """
    paths = glob.glob(pattern)

    paths = [path for path in paths if output_tag not in path]

    paths = sorted(paths, key=natural_sort_key)

    if len(paths) == 0:
        raise RuntimeError("No tables matched: {}".format(pattern))

    return paths


def make_output_path(input_path, output_tag):
    """
    Build the output table path for one input table.

    Example (output_tag="_smooth"):

        input.B0  ->  input_smooth.B0
    """
    if input_path.endswith(".B0"):
        return input_path[:-3] + output_tag + ".B0"

    return input_path + output_tag


# ============================================================
# Mask helpers
# ============================================================

def find_valid_segments(valid):
    """
    Return contiguous True regions of a boolean array as
    (start, end) pairs. `end` is exclusive.
    """
    segments = []

    in_segment = False
    start = None

    for channel in range(len(valid)):
        if valid[channel] and not in_segment:
            start = channel
            in_segment = True

        elif not valid[channel] and in_segment:
            segments.append((start, channel))
            in_segment = False

    if in_segment:
        segments.append((start, len(valid)))

    return segments


def choose_window(requested_window, segment_length):
    """
    Choose an odd window size no larger than the segment it will be
    applied to.
    """
    window = min(requested_window, segment_length)

    if window % 2 == 0:
        window -= 1

    return max(window, 1)


# ============================================================
# Smoothing
# ============================================================

def smooth_real_array(values, valid, window):
    """
    Median-smooth a real-valued spectrum inside each contiguous
    valid region. Flagged gaps are never crossed.
    """
    output = np.full_like(values, np.nan, dtype=float)

    for start, end in find_valid_segments(valid):
        segment = values[start:end]

        if len(segment) == 0:
            continue

        local_window = choose_window(window, len(segment))

        output[start:end] = median_filter(
            segment, size=local_window, mode="nearest"
        )

    return output


def smooth_complex_bandpass(complex_bandpass, flags, window):
    """
    Smooth one complex bandpass spectrum.

    The smoothing is performed on log(amplitude) and unwrapped
    phase. Existing flagged channels remain flagged and are not
    used as smoothing input.
    """
    complex_bandpass = np.asarray(complex_bandpass)
    flags = np.asarray(flags, dtype=bool)

    amplitude = np.abs(complex_bandpass)
    wrapped_phase = np.angle(complex_bandpass)

    valid = (~flags) & np.isfinite(complex_bandpass) & (amplitude > 0.0)

    log_amplitude = np.full_like(amplitude, np.nan, dtype=float)
    log_amplitude[valid] = np.log(amplitude[valid])

    unwrapped_phase = np.full_like(wrapped_phase, np.nan, dtype=float)

    for start, end in find_valid_segments(valid):
        unwrapped_phase[start:end] = np.unwrap(wrapped_phase[start:end])

    smooth_log_amplitude = smooth_real_array(log_amplitude, valid, window)
    smooth_phase = smooth_real_array(unwrapped_phase, valid, window)

    output = complex_bandpass.copy()

    smooth_valid = (
        valid
        & np.isfinite(smooth_log_amplitude)
        & np.isfinite(smooth_phase)
    )

    output[smooth_valid] = np.exp(
        smooth_log_amplitude[smooth_valid] + 1j * smooth_phase[smooth_valid]
    )

    return output


# ============================================================
# Diagnostic plotting
# ============================================================

def plot_table(input_path, output_path, input_amplitudes, input_phases,
                output_amplitudes, output_phases, flags):
    """
    Make a simple input/output diagnostic plot for one table.
    """
    import matplotlib.pyplot as plt

    input_amplitudes = np.asarray(input_amplitudes)
    input_phases = np.asarray(input_phases)
    output_amplitudes = np.asarray(output_amplitudes)
    output_phases = np.asarray(output_phases)
    flags = np.asarray(flags, dtype=bool)

    input_amp_plot = input_amplitudes.copy()
    input_phase_plot = input_phases.copy()
    output_amp_plot = output_amplitudes.copy()
    output_phase_plot = output_phases.copy()

    input_amp_plot[flags] = np.nan
    input_phase_plot[flags] = np.nan
    output_amp_plot[flags] = np.nan
    output_phase_plot[flags] = np.nan

    channels = np.arange(input_amplitudes.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)

    for spectrum in input_amp_plot:
        axes[0, 0].plot(channels, spectrum, linewidth=0.7, alpha=0.7)
    axes[0, 0].set_title("Input amplitude")
    axes[0, 0].set_ylabel("Amplitude")

    for spectrum in input_phase_plot:
        axes[0, 1].plot(channels, spectrum, linewidth=0.7, alpha=0.7)
    axes[0, 1].set_title("Input phase")
    axes[0, 1].set_ylabel("Phase [deg]")

    for spectrum in output_amp_plot:
        axes[1, 0].plot(channels, spectrum, linewidth=0.8)
    axes[1, 0].set_title("Smoothed amplitude")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("Amplitude")

    for spectrum in output_phase_plot:
        axes[1, 1].plot(channels, spectrum, linewidth=0.8)
    axes[1, 1].set_title("Smoothed phase")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Phase [deg]")

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.suptitle(os.path.basename(input_path))
    plt.tight_layout()

    plot_path = output_path + ".png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Diagnostic plot:", plot_path)


# ============================================================
# Process one CASA bandpass table
# ============================================================

def process_table(input_path, output_path, window, make_plots=True):
    """
    Copy one CASA bandpass table and smooth every row/polarization.
    """
    from casacore.tables import table

    print()
    print("Input :", input_path)
    print("Output:", output_path)

    if os.path.exists(output_path):
        print("Removing existing output table")
        shutil.rmtree(output_path)

    shutil.copytree(input_path, output_path)

    tb = table(output_path, readonly=False, ack=False)

    input_amplitudes = []
    input_phases = []
    output_amplitudes = []
    output_phases = []
    plot_flags = []

    try:
        for row in range(tb.nrows()):
            cparam = np.asarray(tb.getcell("CPARAM", row))
            flags = np.asarray(tb.getcell("FLAG", row))

            for pol, input_spectrum, input_flag in iter_pol_spectra(cparam, flags):
                output_spectrum = smooth_complex_bandpass(
                    input_spectrum, input_flag, window
                )

                set_pol_spectrum(cparam, pol, output_spectrum)

                if make_plots:
                    input_amplitudes.append(np.abs(input_spectrum))
                    input_phases.append(np.degrees(np.angle(input_spectrum)))
                    output_amplitudes.append(np.abs(output_spectrum))
                    output_phases.append(np.degrees(np.angle(output_spectrum)))
                    plot_flags.append(input_flag)

            tb.putcell("CPARAM", row, cparam)
            # FLAG is deliberately left unchanged.
    finally:
        tb.flush()
        tb.close()

    if make_plots:
        plot_table(
            input_path=input_path,
            output_path=output_path,
            input_amplitudes=np.array(input_amplitudes),
            input_phases=np.array(input_phases),
            output_amplitudes=np.array(output_amplitudes),
            output_phases=np.array(output_phases),
            flags=np.array(plot_flags),
        )


# ============================================================
# Orchestration (called by the CLI)
# ============================================================

def smooth_tables(input_pattern, output_tag="_smooth", window=95, make_plots=True):
    """
    Smooth every bandpass table matching `input_pattern`, writing a
    new table alongside each input with `output_tag` inserted before
    the `.B0` suffix.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")

    input_tables = find_input_tables(input_pattern, output_tag)

    print("Found {} input tables:".format(len(input_tables)))
    for index, path in enumerate(input_tables):
        print("{:3d}  {}".format(index, path))

    output_paths = []

    for input_path in input_tables:
        output_path = make_output_path(input_path, output_tag)

        process_table(input_path, output_path, window, make_plots=make_plots)

        output_paths.append(output_path)

    print()
    print("Finished.")

    return output_paths
