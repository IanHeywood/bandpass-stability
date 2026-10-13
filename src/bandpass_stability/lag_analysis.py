"""
Read the antenna mapping CSV produced by `antenna_mapping.py` and
calculate adjacent table-to-table bandpass residuals for each
physical antenna.

For each antenna:
  - select the requested SPW and polarization;
  - sort observations by time;
  - time-average the bandpass rows within each table;
  - optionally flag channels in individual epochs that deviate from
    the other epochs in a way that is local in frequency
    (time-direction clipping);
  - compare each available observation with the next one, clipping
    outlier channels against a local running-median baseline;
  - exclude channels that are clipped in a large fraction of pairs
    from every pair, so the per-pair numbers share a channel set;
  - write per-pair and per-epoch metrics and diagnostic plots.
"""

import csv
import os
import warnings

import numpy as np

from .caltable_utils import get_pol_spectrum, safe_filename
from .robust import (
    MAX_CLIP_ITERATIONS,
    circular_centre,
    estimate_linear_phase,
    iterative_stack_clip,
    quiet_nanmedian,
    wrap_phase,
)

PAIR_METRICS_FIELDNAMES = [
    "pair_index",
    "first_table",
    "second_table",
    "first_antenna_index",
    "second_antenna_index",
    "first_time",
    "second_time",
    "dt_seconds",
    "valid_channels",
    "amplitude_channels_clipped",
    "phase_channels_clipped",
    "amplitude_channels_used",
    "phase_channels_used",
    "amplitude_rms_log",
    "amplitude_rms_percent_approx",
    "amplitude_mean_percent_approx",
    "amplitude_std_percent_approx",
    "phase_rms_degrees",
    "phase_mean_degrees",
    "phase_std_degrees",
    "delay_ns",
    "phase_delay_removed_rms_degrees",
]

EPOCH_METRICS_FIELDNAMES = [
    "table",
    "antenna_index",
    "time",
    "valid_channels",
    "time_clipped_channels",
]

# Defaults shared with the CLI.
DEFAULT_CLIP_SIGMA = 5.0
DEFAULT_CLIP_WINDOW = 95
DEFAULT_TIME_CLIP_SIGMA = 5.0
DEFAULT_MASK_FRACTION = 0.25

# The time-direction clip baseline is this many times wider than the
# frequency clip window, so that only features local in frequency are
# rejected and broadband single-epoch changes are kept as signal.
TIME_CLIP_WINDOW_FACTOR = 4


# ============================================================
# Mapping CSV
# ============================================================

def read_mapping_csv(path):
    """
    Read and type-convert the mapping CSV produced by
    `antenna_mapping.build_mapping`.
    """
    rows = []

    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required = {
            "bandpass_table", "mean_time", "antenna_index",
            "antenna_name", "spw", "npol",
        }
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                "Mapping CSV is missing columns: {}".format(sorted(missing))
            )

        for row in reader:
            rows.append(
                {
                    "bandpass_table": row["bandpass_table"],
                    "parent_ms": row.get("parent_ms", ""),
                    "mean_time": float(row["mean_time"]),
                    "antenna_index": int(row["antenna_index"]),
                    "antenna_name": row["antenna_name"],
                    "spw": int(row["spw"]),
                    "npol": int(row["npol"]),
                }
            )

    return rows


# ============================================================
# Read and time-average one table
# ============================================================

# Per-table metadata (ANTENNA1, SPECTRAL_WINDOW_ID and TIME columns,
# plus channel frequencies), read once per table rather than once per
# antenna. Only these small columns are cached; CPARAM/FLAG cells are
# still read per antenna, so memory use does not grow with the number
# of antennas or channels.
_TABLE_METADATA = {}


def read_table_metadata(bandpass_path):
    """
    Return (and cache) the small per-row columns of a bandpass table
    and the channel frequencies of each spectral window.
    """
    if bandpass_path in _TABLE_METADATA:
        return _TABLE_METADATA[bandpass_path]

    from casacore.tables import table

    bp_tb = table(bandpass_path, readonly=True, ack=False)

    try:
        metadata = {
            "antenna": np.asarray(bp_tb.getcol("ANTENNA1")),
            "spw": np.asarray(bp_tb.getcol("SPECTRAL_WINDOW_ID")),
            "time": np.asarray(bp_tb.getcol("TIME"), dtype=float),
        }
    finally:
        bp_tb.close()

    metadata["chan_freq"] = {}

    spw_path = os.path.join(bandpass_path, "SPECTRAL_WINDOW")

    if os.path.isdir(spw_path):
        try:
            spw_tb = table(spw_path, readonly=True, ack=False)
            try:
                for spw_id in range(spw_tb.nrows()):
                    metadata["chan_freq"][spw_id] = np.asarray(
                        spw_tb.getcell("CHAN_FREQ", spw_id), dtype=float
                    )
            finally:
                spw_tb.close()
        except Exception as error:  # frequencies are optional
            print("  could not read channel frequencies from {}: {}".format(
                spw_path, error))

    _TABLE_METADATA[bandpass_path] = metadata

    return metadata


def channel_width_hz(bandpass_path, spw):
    """
    Median channel width in Hz for one SPW, or None if the table has
    no usable SPECTRAL_WINDOW subtable.
    """
    freqs = read_table_metadata(bandpass_path)["chan_freq"].get(spw)

    if freqs is None or freqs.size < 2:
        return None

    return float(np.median(np.abs(np.diff(freqs))))


def read_time_averaged_bandpass(bandpass_path, antenna_index, spw, pol):
    """
    Read all rows for one local antenna index/SPW/polarization and
    average the valid complex solutions over time.
    """
    from casacore.tables import table

    metadata = read_table_metadata(bandpass_path)

    rows = np.where(
        (metadata["antenna"] == antenna_index) & (metadata["spw"] == spw)
    )[0]

    if len(rows) == 0:
        raise RuntimeError(
            "No rows for antenna index {} and SPW {} in {}".format(
                antenna_index, spw, bandpass_path
            )
        )

    rows = rows[np.argsort(metadata["time"][rows])]

    spectra = []
    flag_spectra = []
    times = []

    bp_tb = table(bandpass_path, readonly=True, ack=False)

    try:
        for row in rows:
            cparam = np.asarray(bp_tb.getcell("CPARAM", int(row)))
            flags = np.asarray(bp_tb.getcell("FLAG", int(row)))

            spectrum, spectrum_flags = get_pol_spectrum(cparam, flags, pol)

            spectra.append(spectrum)
            flag_spectra.append(spectrum_flags)
            times.append(metadata["time"][row])
    finally:
        bp_tb.close()

    return average_bandpass_rows(spectra, flag_spectra, times)


def average_bandpass_rows(spectra, flag_spectra, times):
    """
    Time-average a set of complex bandpass spectra, ignoring flagged
    or non-finite samples on a per-channel basis.

    Pulled out from `read_time_averaged_bandpass` so it can be unit
    tested without a real calibration table.
    """
    spectra = np.asarray(spectra)
    flag_spectra = np.asarray(flag_spectra, dtype=bool)
    times = np.asarray(times, dtype=float)

    valid = (
        (~flag_spectra)
        & np.isfinite(spectra)
        & (np.abs(spectra) > 0.0)
    )

    spectra_for_average = spectra.astype(complex)
    spectra_for_average[~valid] = np.nan + 1j * np.nan

    # A fully flagged channel is expected; it becomes NaN and is
    # flagged below, so the "Mean of empty slice" warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        averaged_bandpass = np.nanmean(spectra_for_average, axis=0)

    averaged_flag = np.sum(valid, axis=0) == 0

    mean_time = float(np.mean(times))

    return mean_time, averaged_bandpass, averaged_flag


# ============================================================
# Time-direction clipping
# ============================================================

def time_direction_clip(records, clip_sigma=DEFAULT_TIME_CLIP_SIGMA,
                        window=TIME_CLIP_WINDOW_FACTOR * DEFAULT_CLIP_WINDOW):
    """
    Flag channels in individual epochs that disagree with the other
    epochs for the same antenna, in a way that is local in frequency.

    For each epoch, the deviation from the per-channel median over
    all epochs is formed (log-amplitude, and phase after removing that
    epoch's own phase offset and delay). Each epoch's deviation
    spectrum is then clipped against a wide running-median baseline
    with a local robust sigma, so that:

      - static bandpass structure cancels in the cross-epoch reference,
        which makes wide features such as a 40-channel 180-degree flip
        at a segment edge visible, unlike single-table clipping;
      - a genuine broadband change in one epoch (wider than roughly a
        quarter of `window`) is followed by the baseline and kept, as
        it is the kind of instability this analysis exists to measure.

    Needs at least three epochs with identical channel counts; with
    fewer, nothing is flagged.

    Updates each record's "flag" in place and returns a list with the
    number of channels newly flagged in each record.
    """
    counts = [0] * len(records)

    if clip_sigma is None or clip_sigma <= 0 or len(records) < 3:
        return counts

    shapes = {record["bandpass"].shape for record in records}

    if len(shapes) != 1:
        print("  time-direction clip skipped: channel counts differ between epochs")
        return counts

    bandpasses = np.vstack([record["bandpass"] for record in records]).astype(complex)
    flags = np.vstack([record["flag"] for record in records]).astype(bool)

    original_valid = (~flags) & np.isfinite(bandpasses) & (np.abs(bandpasses) > 0.0)
    keep = original_valid.copy()

    channel = np.arange(bandpasses.shape[1])

    # Deviations from the cross-epoch reference are roughly flat across
    # flagged gaps, so the clip baseline spans the whole band (using
    # only kept channels): treating each flagged gap as a segment edge
    # would let a block bounded by flags become its own baseline and
    # vanish.

    with np.errstate(divide="ignore", invalid="ignore"):
        log_amplitude = np.log(np.abs(bandpasses))
    phase = np.angle(bandpasses)

    for _ in range(MAX_CLIP_ITERATIONS):
        masked_log_amp = np.where(keep, log_amplitude, np.nan)
        reference_amp = quiet_nanmedian(masked_log_amp, axis=0)

        phasors = np.where(keep, np.exp(1j * phase), np.nan)
        reference_phase = np.angle(
            quiet_nanmedian(phasors.real, axis=0) + 1j * quiet_nanmedian(phasors.imag, axis=0)
        )

        amp_deviation = log_amplitude - reference_amp
        phase_deviation = wrap_phase(phase - reference_phase)

        # Remove each epoch's own phase offset and delay relative to
        # the reference, so a delay change cannot masquerade as a
        # frequency-local deviation once wrapped.
        for epoch in range(len(records)):
            if np.count_nonzero(keep[epoch]) >= 3:
                offset, slope = estimate_linear_phase(phase_deviation[epoch], keep[epoch])
                phase_deviation[epoch] = wrap_phase(
                    phase_deviation[epoch] - offset - slope * channel
                )

        new_keep = (
            iterative_stack_clip(amp_deviation, keep, window, clip_sigma)
            & iterative_stack_clip(phase_deviation, keep, window, clip_sigma, wrap=True)
        )

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep

    for epoch, record in enumerate(records):
        newly_flagged = original_valid[epoch] & ~keep[epoch]
        counts[epoch] = int(np.count_nonzero(newly_flagged))
        record["flag"] = np.asarray(record["flag"], dtype=bool) | newly_flagged

    return counts


# ============================================================
# Adjacent residuals
# ============================================================

def pair_residuals(first, second):
    """
    Form the log-amplitude and phase residuals of one adjacent pair.

    The phase residual is returned relative to its own circular
    centre, so a residual sitting near +/-180 degrees is not split in
    two by wrapping. Adding `centre` back gives the residual relative
    to zero.

    Returns (log_amplitude, centred_phase, centre, valid), with NaN at
    invalid channels.
    """
    bp1 = first["bandpass"]
    bp2 = second["bandpass"]

    valid = (
        (~first["flag"]) & (~second["flag"])
        & np.isfinite(bp1) & np.isfinite(bp2)
        & (np.abs(bp1) > 0.0) & (np.abs(bp2) > 0.0)
    )

    log_amplitude = np.full(bp1.shape, np.nan, dtype=float)
    raw_phase = np.full(bp1.shape, np.nan, dtype=float)

    ratio = bp2[valid] / bp1[valid]

    log_amplitude[valid] = np.log(np.abs(ratio))
    raw_phase[valid] = np.angle(ratio)

    centre = circular_centre(raw_phase, valid)

    centred_phase = np.full(bp1.shape, np.nan, dtype=float)
    centred_phase[valid] = wrap_phase(raw_phase[valid] - centre)

    return log_amplitude, centred_phase, centre, valid


def pair_metrics(log_amplitude, centred_phase, centre, amplitude_keep,
                 phase_keep, chan_width_hz=None):
    """
    Summary statistics for one pair over the kept channels.

    Amplitude and phase RMS are about zero, so they include any
    overall offset between the epochs; the mean and standard deviation
    split that into a constant shift and the channel-to-channel
    scatter. The delay-removed phase RMS additionally removes a linear
    phase slope across the band (fitted robustly, including slopes
    that wrap several times), and `delay_ns` is that slope as a delay
    if the channel width is known.
    """
    nan = float("nan")

    amplitude = log_amplitude[amplitude_keep]
    phase = centred_phase[phase_keep] + centre

    metrics = {
        "amplitude_rms_log": nan,
        "amplitude_mean_log": nan,
        "amplitude_std_log": nan,
        "phase_rms_degrees": nan,
        "phase_mean_degrees": nan,
        "phase_std_degrees": nan,
        "delay_ns": nan,
        "phase_delay_removed_rms_degrees": nan,
    }

    if amplitude.size:
        metrics["amplitude_rms_log"] = float(np.sqrt(np.mean(amplitude ** 2)))
        metrics["amplitude_mean_log"] = float(np.mean(amplitude))
        metrics["amplitude_std_log"] = float(np.std(amplitude))

    if phase.size:
        metrics["phase_rms_degrees"] = float(np.degrees(np.sqrt(np.mean(phase ** 2))))
        metrics["phase_mean_degrees"] = float(np.degrees(np.mean(phase)))
        metrics["phase_std_degrees"] = float(np.degrees(np.std(phase)))

        channel = np.arange(centred_phase.size)
        offset, slope = estimate_linear_phase(centred_phase, phase_keep)
        delay_removed = wrap_phase(centred_phase - offset - slope * channel)[phase_keep]

        metrics["phase_delay_removed_rms_degrees"] = float(
            np.degrees(np.sqrt(np.mean(delay_removed ** 2)))
        )

        if chan_width_hz:
            metrics["delay_ns"] = float(slope / (2.0 * np.pi * chan_width_hz) * 1e9)

    return metrics


def persistent_channel_mask(valid_list, keep_list, mask_fraction):
    """
    Channels clipped in more than `mask_fraction` of the pairs in
    which they were valid. Returns a boolean mask (True = exclude).
    """
    if mask_fraction is None or mask_fraction >= 1 or not valid_list:
        return np.zeros(valid_list[0].shape if valid_list else 0, dtype=bool)

    valid = np.vstack(valid_list)
    clipped = valid & ~np.vstack(keep_list)

    valid_count = np.sum(valid, axis=0)
    clipped_count = np.sum(clipped, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(valid_count > 0, clipped_count / valid_count, 0.0)

    return fraction > mask_fraction


def calculate_adjacent_residuals(records, max_pair_gap_seconds=None,
                                 clip_sigma=DEFAULT_CLIP_SIGMA,
                                 clip_window=DEFAULT_CLIP_WINDOW,
                                 mask_fraction=DEFAULT_MASK_FRACTION,
                                 chan_width_hz=None):
    """
    Compare each available observation with the next observation for
    the same physical antenna.

    For each pair, outlier channels are iteratively clipped
    (independently for amplitude and phase) against a running-median
    baseline along frequency, with a local robust sigma as the noise
    scale, so RFI-like spikes do not dominate the headline numbers
    while smooth genuine structure is kept. The count of clipped
    channels is reported per pair, so RFI severity over time stays
    visible rather than silently disappearing.

    Channels clipped in more than `mask_fraction` of pairs are then
    excluded from every pair, so the per-pair numbers are computed
    over a common channel set apart from transient outliers and flags.

    `records` is a chronologically-sorted list of dicts with keys
    "bandpass_table", "antenna_index", "time", "bandpass", "flag".

    Returns (pair_rows, amplitude_residuals, phase_residuals): the
    per-pair metric dicts, plus stacked per-channel residual arrays
    (one row per pair; log-amplitude, and phase in radians relative to
    zero), NaN at invalid, clipped and masked channels.
    """
    pairs = []

    for index in range(len(records) - 1):
        first = records[index]
        second = records[index + 1]

        dt_seconds = second["time"] - first["time"]

        if (
            max_pair_gap_seconds is not None
            and dt_seconds > max_pair_gap_seconds
        ):
            print("  skipping pair with gap {:.1f} s".format(dt_seconds))
            continue

        if first["bandpass"].shape != second["bandpass"].shape:
            print(
                "  skipping pair because channel counts differ: "
                "{} versus {}".format(first["bandpass"].shape, second["bandpass"].shape)
            )
            continue

        log_amplitude, centred_phase, centre, valid = pair_residuals(first, second)

        if not valid.any():
            print("  skipping pair with no common valid channels")
            continue

        pairs.append(
            {
                "first": first,
                "second": second,
                "dt_seconds": dt_seconds,
                "log_amplitude": log_amplitude,
                "centred_phase": centred_phase,
                "centre": centre,
                "valid": valid,
            }
        )

    if not pairs:
        return [], np.empty((0, 0)), np.empty((0, 0))

    # Clip all pairs together: each pair against its own running
    # baseline along frequency (spanning flagged gaps, as the residuals
    # are roughly continuous across them, so a feature next to a gap
    # is judged against both sides), with the noise scale at each
    # channel measured across pairs. Pairs are grouped by channel
    # count, which in practice is a single group.
    for shape in {pair["valid"].shape for pair in pairs}:
        group = [pair for pair in pairs if pair["valid"].shape == shape]
        valid_stack = np.vstack([pair["valid"] for pair in group])

        amplitude_keep = iterative_stack_clip(
            np.vstack([pair["log_amplitude"] for pair in group]),
            valid_stack, clip_window, clip_sigma,
        )
        # Remove each pair's delay slope before clipping, so a delay
        # change between epochs cannot look like a local outlier.
        derotated_phase = []
        for pair in group:
            channel = np.arange(pair["centred_phase"].size)
            offset, slope = estimate_linear_phase(pair["centred_phase"], pair["valid"])
            derotated_phase.append(wrap_phase(pair["centred_phase"] - offset - slope * channel))

        phase_keep = iterative_stack_clip(
            np.vstack(derotated_phase),
            valid_stack, clip_window, clip_sigma, wrap=True,
        )

        for pair, amp_row, phase_row in zip(group, amplitude_keep, phase_keep):
            pair["amplitude_keep"] = amp_row
            pair["phase_keep"] = phase_row

    if len({pair["valid"].shape for pair in pairs}) == 1:
        amplitude_mask = persistent_channel_mask(
            [pair["valid"] for pair in pairs],
            [pair["amplitude_keep"] for pair in pairs],
            mask_fraction,
        )
        phase_mask = persistent_channel_mask(
            [pair["valid"] for pair in pairs],
            [pair["phase_keep"] for pair in pairs],
            mask_fraction,
        )
    else:
        print("  common channel mask skipped: channel counts differ between pairs")
        amplitude_mask = phase_mask = np.zeros(0, dtype=bool)

    if amplitude_mask.any() or phase_mask.any():
        print(
            "  channels masked in all pairs (clipped in > {:.0%} of pairs): "
            "{} amplitude, {} phase".format(
                mask_fraction, int(amplitude_mask.sum()), int(phase_mask.sum())
            )
        )

    pair_rows = []
    amplitude_residuals = []
    phase_residuals = []

    for pair in pairs:
        valid = pair["valid"]
        amplitude_used = pair["amplitude_keep"].copy()
        phase_used = pair["phase_keep"].copy()

        if amplitude_mask.shape == valid.shape:
            amplitude_used &= ~amplitude_mask
            phase_used &= ~phase_mask

        metrics = pair_metrics(
            pair["log_amplitude"], pair["centred_phase"], pair["centre"],
            amplitude_used, phase_used, chan_width_hz=chan_width_hz,
        )

        pair_rows.append(
            {
                "pair_index": len(pair_rows),
                "first_table": pair["first"]["bandpass_table"],
                "second_table": pair["second"]["bandpass_table"],
                "first_antenna_index": pair["first"]["antenna_index"],
                "second_antenna_index": pair["second"]["antenna_index"],
                "first_time": pair["first"]["time"],
                "second_time": pair["second"]["time"],
                "dt_seconds": pair["dt_seconds"],
                "valid_channels": int(np.count_nonzero(valid)),
                "amplitude_channels_clipped": int(np.count_nonzero(valid & ~pair["amplitude_keep"])),
                "phase_channels_clipped": int(np.count_nonzero(valid & ~pair["phase_keep"])),
                "amplitude_channels_used": int(np.count_nonzero(amplitude_used)),
                "phase_channels_used": int(np.count_nonzero(phase_used)),
                "amplitude_rms_log": metrics["amplitude_rms_log"],
                "amplitude_rms_percent_approx": 100.0 * metrics["amplitude_rms_log"],
                "amplitude_mean_percent_approx": 100.0 * metrics["amplitude_mean_log"],
                "amplitude_std_percent_approx": 100.0 * metrics["amplitude_std_log"],
                "phase_rms_degrees": metrics["phase_rms_degrees"],
                "phase_mean_degrees": metrics["phase_mean_degrees"],
                "phase_std_degrees": metrics["phase_std_degrees"],
                "delay_ns": metrics["delay_ns"],
                "phase_delay_removed_rms_degrees": metrics["phase_delay_removed_rms_degrees"],
            }
        )

        amplitude_for_plot = np.where(amplitude_used, pair["log_amplitude"], np.nan)
        phase_for_plot = np.where(phase_used, pair["centred_phase"] + pair["centre"], np.nan)

        amplitude_residuals.append(amplitude_for_plot)
        phase_residuals.append(phase_for_plot)

    return (
        pair_rows,
        np.asarray(amplitude_residuals),
        np.asarray(phase_residuals),
    )


# ============================================================
# Outputs
# ============================================================

def write_csv(path, fieldnames, rows):
    """
    Write a list of dicts to CSV.
    """
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pair_metrics(path, pair_rows):
    """
    Write one CSV row per adjacent table pair.
    """
    write_csv(path, PAIR_METRICS_FIELDNAMES, pair_rows)


def _relative_time_axis(relative_seconds):
    """
    Convert an array of seconds-since-reference into whichever unit
    (minutes/hours/days) keeps the numbers readable, based on the
    total span covered.
    """
    relative_seconds = np.asarray(relative_seconds, dtype=float)

    if len(relative_seconds) == 0:
        return relative_seconds, "s"

    span = float(np.max(relative_seconds) - np.min(relative_seconds))

    if span >= 2 * 86400:
        return relative_seconds / 86400.0, "days"

    if span >= 2 * 3600:
        return relative_seconds / 3600.0, "hours"

    return relative_seconds / 60.0, "minutes"


def plot_pair_metrics(path, antenna_name, pair_rows, reference_time):
    """
    Plot the per-pair amplitude and phase residual statistics against
    real elapsed time (the midpoint of each pair's two epochs,
    relative to `reference_time`, the antenna's first observation).

    Solid lines are the RMS about zero (the headline metric); dashed
    lines remove the mean offset, and for phase the dotted line also
    removes a delay slope.
    """
    import matplotlib.pyplot as plt

    first_times = np.array([row["first_time"] for row in pair_rows])
    second_times = np.array([row["second_time"] for row in pair_rows])

    # Get the mid-point
    representative_times = 0.5 * (first_times + second_times)
    relative_seconds = representative_times - reference_time

    time_values, time_unit = _relative_time_axis(relative_seconds)

    def column(name):
        return np.array([row[name] for row in pair_rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(time_values, column("amplitude_rms_percent_approx"), marker="o",
                 markerfacecolor="skyblue", markeredgecolor="black", color="skyblue",
                 label="RMS")
    axes[0].plot(time_values, column("amplitude_std_percent_approx"), marker="s",
                 markersize=4, linestyle="--", color="steelblue",
                 label="mean removed")
    axes[0].set_ylabel("Amplitude RMS [%]")
    axes[0].set_title("{}: adjacent-table amplitude residual".format(antenna_name))

    axes[1].plot(time_values, column("phase_rms_degrees"), marker="o",
                 markerfacecolor="hotpink", markeredgecolor="black", color="hotpink",
                 label="RMS")
    axes[1].plot(time_values, column("phase_std_degrees"), marker="s",
                 markersize=4, linestyle="--", color="mediumvioletred",
                 label="mean removed")
    axes[1].plot(time_values, column("phase_delay_removed_rms_degrees"), marker="^",
                 markersize=4, linestyle=":", color="purple",
                 label="mean and delay removed")
    axes[1].set_xlabel("Time relative to first epoch [{}]".format(time_unit))
    axes[1].set_ylabel("Phase RMS [deg]")
    axes[1].set_title("{}: adjacent-table phase residual".format(antenna_name))

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_channel_summary(path, antenna_name, amplitude, phase):
    """
    Plot the median and middle 68 percent of adjacent-pair residuals
    for each channel.
    """
    import matplotlib.pyplot as plt

    channels = np.arange(amplitude.shape[1])

    # Channels that are NaN in every pair (flagged or masked) are
    # expected, so silence the all-NaN-slice warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        amplitude_median = np.nanmedian(amplitude, axis=0)
        amplitude_low = np.nanpercentile(amplitude, 16, axis=0)
        amplitude_high = np.nanpercentile(amplitude, 84, axis=0)

        phase_degrees = np.degrees(phase)
        phase_median = np.nanmedian(phase_degrees, axis=0)
        phase_low = np.nanpercentile(phase_degrees, 16, axis=0)
        phase_high = np.nanpercentile(phase_degrees, 84, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(channels, 100.0 * amplitude_median, color='skyblue')
    axes[0].fill_between(
        channels, 100.0 * amplitude_low, 100.0 * amplitude_high, alpha=0.3, color='skyblue'
    )
    axes[0].axhline(0.0, linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Amplitude residual [%]")
    axes[0].set_title("{}: residual per channel".format(antenna_name))
    axes[0].tick_params(axis='both', direction='in', length=6, color='black')

    axes[1].plot(channels, phase_median, color='hotpink')
    axes[1].fill_between(channels, phase_low, phase_high, alpha=0.3, color='hotpink')
    axes[1].axhline(0.0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Phase residual [deg]")
    axes[1].tick_params(axis='both', direction='in', length=6, color='black')

    for axis in axes:
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# Process one physical antenna
# ============================================================

def process_antenna(antenna_name, mapping_rows, output_directory, spw, pol,
                    max_pair_gap_seconds=None, make_plots=True,
                    clip_sigma=DEFAULT_CLIP_SIGMA, clip_window=DEFAULT_CLIP_WINDOW,
                    time_clip_sigma=DEFAULT_TIME_CLIP_SIGMA,
                    mask_fraction=DEFAULT_MASK_FRACTION):
    """
    Build the chronological time series for one physical antenna and
    produce adjacent-pair metrics.
    """
    print()
    print("Processing antenna:", antenna_name)

    records = []

    for mapping_row in sorted(mapping_rows, key=lambda row: row["mean_time"]):
        if mapping_row["npol"] <= pol:
            print(
                "  skipping {} because POL {} is unavailable".format(
                    mapping_row["bandpass_table"], pol
                )
            )
            continue

        mean_time, bandpass, flag = read_time_averaged_bandpass(
            mapping_row["bandpass_table"], mapping_row["antenna_index"], spw, pol
        )

        records.append(
            {
                "bandpass_table": mapping_row["bandpass_table"],
                "antenna_index": mapping_row["antenna_index"],
                "time": mean_time,
                "bandpass": bandpass,
                "flag": flag,
            }
        )

        print(
            "  {:s}: local index {:d}, time {:.3f}".format(
                os.path.basename(mapping_row["bandpass_table"].rstrip(os.sep)),
                mapping_row["antenna_index"], mean_time
            )
        )

    if len(records) < 2:
        print("  fewer than two usable observations; skipping")
        return

    records.sort(key=lambda record: record["time"])

    time_clipped = time_direction_clip(
        records, clip_sigma=time_clip_sigma,
        window=TIME_CLIP_WINDOW_FACTOR * clip_window,
    )

    if any(time_clipped):
        print("  time-direction clip flagged {} channels across {} epochs".format(
            sum(time_clipped), sum(1 for count in time_clipped if count)))

    pair_rows, amplitude_residuals, phase_residuals = calculate_adjacent_residuals(
        records, max_pair_gap_seconds=max_pair_gap_seconds,
        clip_sigma=clip_sigma, clip_window=clip_window,
        mask_fraction=mask_fraction,
        chan_width_hz=channel_width_hz(records[0]["bandpass_table"], spw),
    )

    if len(pair_rows) == 0:
        print("  no valid adjacent pairs; skipping outputs")
        return

    file_stem = safe_filename(antenna_name)

    antenna_directory = os.path.join(output_directory, file_stem)
    os.makedirs(antenna_directory, exist_ok=True)

    metrics_path = os.path.join(antenna_directory, "{}_pair_metrics.csv".format(file_stem))
    write_pair_metrics(metrics_path, pair_rows)

    epoch_rows = [
        {
            "table": record["bandpass_table"],
            "antenna_index": record["antenna_index"],
            "time": record["time"],
            "valid_channels": int(np.count_nonzero(
                ~record["flag"] & np.isfinite(record["bandpass"])
            )),
            "time_clipped_channels": count,
        }
        for record, count in zip(records, time_clipped)
    ]
    write_csv(
        os.path.join(antenna_directory, "{}_epoch_metrics.csv".format(file_stem)),
        EPOCH_METRICS_FIELDNAMES, epoch_rows,
    )

    if make_plots:
        pair_plot_path = os.path.join(antenna_directory, "{}_pair_rms.png".format(file_stem))
        channel_plot_path = os.path.join(
            antenna_directory, "{}_channel_summary.png".format(file_stem)
        )

        plot_pair_metrics(pair_plot_path, antenna_name, pair_rows, records[0]["time"])
        plot_channel_summary(
            channel_plot_path, antenna_name, amplitude_residuals, phase_residuals
        )

    print("  adjacent pairs:", len(pair_rows))
    print(
        "  median amplitude RMS [% approx.]: {:.6f}".format(
            quiet_nanmedian(
                [row["amplitude_rms_percent_approx"] for row in pair_rows]
            )
        )
    )
    print(
        "  median phase RMS [deg]: {:.6f}".format(
            quiet_nanmedian([row["phase_rms_degrees"] for row in pair_rows])
        )
    )
    print("  wrote:", antenna_directory)


# ============================================================
# Orchestration (called by the CLI)
# ============================================================

def run_lag_analysis(mapping_csv, output_directory="lag_residual_results",
                     spw=0, pol=0, only_antenna=None,
                     max_pair_gap_seconds=None, make_plots=True,
                     clip_sigma=DEFAULT_CLIP_SIGMA, clip_window=DEFAULT_CLIP_WINDOW,
                     time_clip_sigma=DEFAULT_TIME_CLIP_SIGMA,
                     mask_fraction=DEFAULT_MASK_FRACTION):
    """
    Run the full adjacent-pair lag analysis for every physical
    antenna present in `mapping_csv` at the given SPW.
    """
    if clip_window < 1 or clip_window % 2 == 0:
        raise ValueError("clip_window must be a positive odd number")

    mapping_rows = read_mapping_csv(mapping_csv)

    selected_rows = [row for row in mapping_rows if row["spw"] == spw]

    if only_antenna is not None:
        selected_rows = [
            row for row in selected_rows if row["antenna_name"] == only_antenna
        ]

    if len(selected_rows) == 0:
        raise RuntimeError(
            "No mapping rows match SPW {} and antenna selection {}".format(
                spw, only_antenna
            )
        )

    antenna_names = sorted(set(row["antenna_name"] for row in selected_rows))

    print(
        "Processing {} physical antennas for SPW {} POL {}".format(
            len(antenna_names), spw, pol
        )
    )

    os.makedirs(output_directory, exist_ok=True)

    for antenna_name in antenna_names:
        antenna_rows = [
            row for row in selected_rows if row["antenna_name"] == antenna_name
        ]

        process_antenna(
            antenna_name, antenna_rows, output_directory, spw, pol,
            max_pair_gap_seconds=max_pair_gap_seconds, make_plots=make_plots,
            clip_sigma=clip_sigma, clip_window=clip_window,
            time_clip_sigma=time_clip_sigma, mask_fraction=mask_fraction,
        )

    print()
    print("Finished. Results are in:", output_directory)

    return output_directory
