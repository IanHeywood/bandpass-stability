"""
De-noise CASA bandpass calibration tables with a median filter along
the frequency axis, optionally flagging errant channels first.

Smoothing is performed on log-amplitude and phase independently,
inside each contiguous run of unflagged channels. Phase is smoothed
without unwrapping (after removing the overall delay slope), so a
short half-turn block cannot shift the rest of the segment by a turn.
Flagged channels are never smoothed across and are left unchanged.

Two optional cleaning steps can run before the smoothing, and both
write their results into the FLAG column of the output table:

  - flag growth: extend every existing flagged region by a number of
    channels on each side, to catch residual RFI in the wings of
    regions that were flagged but not quite widely enough;
  - local clipping: iteratively flag channels whose log-amplitude or
    phase deviates from a running-median baseline by more than a
    chosen multiple of the local robust sigma.

The running median mirrors the data at segment edges
(`mode="reflect"`). Padding with copies of the edge channel instead
(`mode="nearest"`) lets a feature at a segment edge win the median,
so it survives smoothing intact and is invisible to clipping.
"""

import glob
import os
import shutil

import numpy as np

from .caltable_utils import iter_pol_spectra, natural_sort_key, set_pol_spectrum
from .robust import (  # noqa: F401  (choose_window, find_valid_segments re-exported)
    choose_window,
    circular_running_median,
    estimate_linear_phase,
    find_valid_segments,
    grow_flags,
    iterative_local_clip,
    masked_running_median,
    wrap_phase,
)


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
    input_path = input_path.rstrip(os.sep)

    if input_path.endswith(".B0"):
        return input_path[:-3] + output_tag + ".B0"

    return input_path + output_tag


# ============================================================
# Smoothing
# ============================================================

def smooth_real_array(values, valid, window, use=None):
    """
    Median-smooth a real-valued spectrum inside each contiguous
    valid region. Flagged gaps are never crossed.

    If `use` is given, only those channels contribute to (and receive)
    the smoothed values, but they are filtered as one run within each
    valid segment, so channels removed by clipping are bridged.
    """
    if use is None:
        use = valid

    return masked_running_median(values, valid, use, window)


def clean_and_smooth_bandpass(complex_bandpass, flags, window,
                              clip_sigma=None, flag_grow=0):
    """
    Optionally flag errant channels, then smooth one complex bandpass
    spectrum.

    Steps:
      1. channels that are flagged, non-finite or zero are invalid;
      2. if `flag_grow` > 0, every invalid region is widened by that
         many channels on each side;
      3. if `clip_sigma` is set, channels whose log-amplitude or phase
         deviates from a running-median baseline by more than
         `clip_sigma` local robust sigmas are rejected, iteratively,
         until a pass rejects nothing new;
      4. the surviving channels are median-smoothed in log-amplitude
         and phase. Phase is smoothed without unwrapping: the overall
         delay slope is removed, a wrap-proof running median is taken
         (`circular_running_median`), and the slope is added back.

    Returns (smoothed_bandpass, output_flags, n_grown, n_clipped).
    Channels flagged in steps 2-3 keep their original (unsmoothed)
    values and are set True in `output_flags`; all other flags are
    passed through unchanged.
    """
    complex_bandpass = np.asarray(complex_bandpass)
    flags = np.asarray(flags, dtype=bool)

    amplitude = np.abs(complex_bandpass)
    wrapped_phase = np.angle(complex_bandpass)

    valid = (~flags) & np.isfinite(complex_bandpass) & (amplitude > 0.0)

    grown_valid = grow_flags(valid, flag_grow)
    n_grown = int(np.count_nonzero(valid & ~grown_valid))

    log_amplitude = np.full(amplitude.shape, np.nan, dtype=float)
    log_amplitude[valid] = np.log(amplitude[valid])

    # Remove the overall delay slope, so the phase changes slowly within
    # one window and can be handled without unwrapping (see
    # `circular_running_median`); it is added back at the end.
    channel = np.arange(complex_bandpass.size)
    _, slope = estimate_linear_phase(wrapped_phase, grown_valid)
    derotated_phase = wrap_phase(wrapped_phase - slope * channel)
    derotated_phase[~valid] = np.nan

    keep = iterative_local_clip(
        np.vstack([log_amplitude, derotated_phase]),
        segments_valid=grown_valid,
        window=window,
        clip_sigma=clip_sigma,
        wrap=[False, True],
        use=grown_valid,
    )

    n_clipped = int(np.count_nonzero(grown_valid & ~keep))

    smooth_log_amplitude = smooth_real_array(log_amplitude, grown_valid, window, use=keep)
    smooth_phase = (
        circular_running_median(derotated_phase, grown_valid, keep, window)
        + slope * channel
    )

    output = complex_bandpass.copy()

    smooth_valid = (
        keep
        & np.isfinite(smooth_log_amplitude)
        & np.isfinite(smooth_phase)
    )

    output[smooth_valid] = np.exp(
        smooth_log_amplitude[smooth_valid] + 1j * smooth_phase[smooth_valid]
    )

    output_flags = flags | (valid & ~keep)

    return output, output_flags, n_grown, n_clipped


def smooth_complex_bandpass(complex_bandpass, flags, window):
    """
    Smooth one complex bandpass spectrum without any extra flagging.

    The smoothing is performed on log(amplitude) and phase. Existing flagged channels remain flagged and are not
    used as smoothing input.
    """
    output, _, _, _ = clean_and_smooth_bandpass(complex_bandpass, flags, window)

    return output


# ============================================================
# Diagnostic plotting
# ============================================================

def plot_table(input_path, output_path, input_amplitudes, input_phases,
               output_amplitudes, output_phases, flags, new_flags=None):
    """
    Make a simple input/output diagnostic plot for one table.

    `flags` are the original flags; `new_flags` (optional) marks
    channels flagged by this step, which are drawn as black crosses on
    the input panels and are absent from the output panels.
    """
    import matplotlib.pyplot as plt

    input_amplitudes = np.asarray(input_amplitudes)
    input_phases = np.asarray(input_phases)
    output_amplitudes = np.asarray(output_amplitudes)
    output_phases = np.asarray(output_phases)
    flags = np.asarray(flags, dtype=bool)

    if new_flags is None:
        new_flags = np.zeros_like(flags)
    new_flags = np.asarray(new_flags, dtype=bool)

    input_amp_plot = input_amplitudes.copy()
    input_phase_plot = input_phases.copy()
    output_amp_plot = output_amplitudes.copy()
    output_phase_plot = output_phases.copy()

    input_amp_plot[flags] = np.nan
    input_phase_plot[flags] = np.nan
    output_amp_plot[flags | new_flags] = np.nan
    output_phase_plot[flags | new_flags] = np.nan

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

    if new_flags.any():
        rows, columns = np.nonzero(new_flags)
        axes[0, 0].plot(
            columns, input_amplitudes[rows, columns], "x", color="black",
            markersize=3, label="flagged by this step",
        )
        axes[0, 1].plot(
            columns, input_phases[rows, columns], "x", color="black",
            markersize=3, label="flagged by this step",
        )
        axes[0, 0].legend(loc="best", fontsize="small")

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

    fig.suptitle(os.path.basename(input_path.rstrip(os.sep)))
    plt.tight_layout()

    plot_path = output_path + ".png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("Diagnostic plot:", plot_path)


# ============================================================
# Process one CASA bandpass table
# ============================================================

def process_table(input_path, output_path, window, make_plots=True,
                  clip_sigma=None, flag_grow=0):
    """
    Copy one CASA bandpass table, optionally flag errant channels,
    and smooth every row/polarization.

    Returns (n_grown, n_clipped, n_solutions): totals over all rows
    and polarizations.
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
    plot_new_flags = []

    total_grown = 0
    total_clipped = 0
    n_solutions = 0

    try:
        for row in range(tb.nrows()):
            cparam = np.asarray(tb.getcell("CPARAM", row))
            flags = np.asarray(tb.getcell("FLAG", row))

            flags_changed = False

            for pol, input_spectrum, input_flag in iter_pol_spectra(cparam, flags):
                output_spectrum, output_flag, n_grown, n_clipped = clean_and_smooth_bandpass(
                    input_spectrum, input_flag, window,
                    clip_sigma=clip_sigma, flag_grow=flag_grow,
                )

                set_pol_spectrum(cparam, pol, output_spectrum)

                if n_grown or n_clipped:
                    set_pol_spectrum(flags, pol, output_flag)
                    flags_changed = True

                total_grown += n_grown
                total_clipped += n_clipped
                n_solutions += 1

                if make_plots:
                    input_amplitudes.append(np.abs(input_spectrum))
                    input_phases.append(np.degrees(np.angle(input_spectrum)))
                    output_amplitudes.append(np.abs(output_spectrum))
                    output_phases.append(np.degrees(np.angle(output_spectrum)))
                    plot_flags.append(input_flag)
                    plot_new_flags.append(output_flag & ~input_flag)

            tb.putcell("CPARAM", row, cparam)

            if flags_changed:
                tb.putcell("FLAG", row, flags)
    finally:
        tb.flush()
        tb.close()

    if flag_grow or clip_sigma:
        print(
            "Flagged by this step: {} channels by flag growth, {} by clipping "
            "(over {} antenna/polarization solutions)".format(
                total_grown, total_clipped, n_solutions
            )
        )

    if make_plots:
        plot_table(
            input_path=input_path,
            output_path=output_path,
            input_amplitudes=np.array(input_amplitudes),
            input_phases=np.array(input_phases),
            output_amplitudes=np.array(output_amplitudes),
            output_phases=np.array(output_phases),
            flags=np.array(plot_flags),
            new_flags=np.array(plot_new_flags),
        )

    return total_grown, total_clipped, n_solutions


# ============================================================
# Orchestration (called by the CLI)
# ============================================================

def smooth_tables(input_pattern, output_tag="_smooth", window=95, make_plots=True,
                  clip_sigma=None, flag_grow=0):
    """
    Smooth every bandpass table matching `input_pattern`, writing a
    new table alongside each input with `output_tag` inserted before
    the `.B0` suffix.

    `clip_sigma` (default None: off) and `flag_grow` (default 0: off)
    enable the optional cleaning steps described in
    `clean_and_smooth_bandpass`.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")

    if flag_grow < 0:
        raise ValueError("flag_grow must be zero or positive")

    input_tables = find_input_tables(input_pattern, output_tag)

    print("Found {} input tables:".format(len(input_tables)))
    for index, path in enumerate(input_tables):
        print("{:3d}  {}".format(index, path))

    output_paths = []

    for input_path in input_tables:
        output_path = make_output_path(input_path, output_tag)

        process_table(
            input_path, output_path, window, make_plots=make_plots,
            clip_sigma=clip_sigma, flag_grow=flag_grow,
        )

        output_paths.append(output_path)

    print()
    print("Finished.")

    return output_paths
