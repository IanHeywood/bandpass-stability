"""
Read the antenna mapping CSV produced by `antenna_mapping.py` and
calculate adjacent table-to-table bandpass residuals for each
physical antenna.

For each antenna:
  - select the requested SPW and polarization;
  - sort observations by time;
  - time-average the bandpass rows within each table;
  - compare each available observation with the next one;
  - write per-pair metrics and diagnostic plots.
"""

import csv
import os

import numpy as np

from .caltable_utils import get_pol_spectrum, safe_filename

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
    "amplitude_rms_log",
    "amplitude_rms_percent_approx",
    "phase_rms_degrees",
]

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

def read_time_averaged_bandpass(bandpass_path, antenna_index, spw, pol):
    """
    Read all rows for one local antenna index/SPW/polarization and
    average the valid complex solutions over time.
    """
    from casacore.tables import table

    bp_tb = table(bandpass_path, readonly=True, ack=False)

    try:
        antenna_column = bp_tb.getcol("ANTENNA1")
        spw_column = bp_tb.getcol("SPECTRAL_WINDOW_ID")
        time_column = bp_tb.getcol("TIME")

        rows = np.where(
            (antenna_column == antenna_index) & (spw_column == spw)
        )[0]

        if len(rows) == 0:
            raise RuntimeError(
                "No rows for antenna index {} and SPW {} in {}".format(
                    antenna_index, spw, bandpass_path
                )
            )

        rows = rows[np.argsort(time_column[rows])]

        spectra = []
        flag_spectra = []
        times = []

        for row in rows:
            cparam = np.asarray(bp_tb.getcell("CPARAM", int(row)))
            flags = np.asarray(bp_tb.getcell("FLAG", int(row)))

            spectrum, spectrum_flags = get_pol_spectrum(cparam, flags, pol)

            spectra.append(spectrum)
            flag_spectra.append(spectrum_flags)
            times.append(time_column[row])
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

    spectra_for_average = spectra.copy()
    spectra_for_average[~valid] = np.nan + 1j * np.nan

    with np.errstate(invalid="ignore"):
        averaged_bandpass = np.nanmean(spectra_for_average, axis=0)

    averaged_flag = np.sum(valid, axis=0) == 0

    mean_time = float(np.mean(times))

    return mean_time, averaged_bandpass, averaged_flag


# ============================================================
# Adjacent residuals
# ============================================================

def robust_sigma_clip_mask(values, n_iterations=5, clip_sigma=5.0):
    """
    Iteratively flag outliers in a real-valued array using a
    median/MAD-based robust sigma, so a handful of RFI-contaminated
    channels don't dominate a downstream RMS.

    Unlike a fixed edge-trim, this adapts to wherever the bad
    channels actually are in a given pair, which matters because RFI
    occupancy shifts from epoch to epoch. A genuine broadband
    instability (a shift across many channels at once) will not get
    clipped, since the robust scale itself would shift with it.

    Returns a boolean mask, True for channels to KEEP.
    """
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)

    # Scale factor so MAD approximates the standard deviation for
    # a Gaussian core distribution.
    mad_to_sigma = 1.4826

    for _ in range(n_iterations):
        if np.count_nonzero(keep) < 3:
            break

        current = values[keep]
        median = np.median(current)
        mad = np.median(np.abs(current - median))

        if mad == 0:
            break

        sigma = mad * mad_to_sigma
        new_keep = keep & (np.abs(values - median) <= clip_sigma * sigma)

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep

    return keep


def calculate_adjacent_residuals(records, max_pair_gap_seconds=None, clip_sigma=5.0):
    """
    Compare each available observation with the next observation for
    the same physical antenna.

    Before computing each pair's RMS, RFI-like outlier channels are
    iteratively sigma-clipped (independently for amplitude and
    phase) so a handful of contaminated channels don't dominate the
    headline stability number. The count of clipped channels is
    reported per pair so RFI severity over time stays visible rather
    than silently disappearing.

    `records` is a chronologically-sorted list of dicts with keys
    "bandpass_table", "antenna_index", "time", "bandpass", "flag".

    Returns a list of per-pair metric dicts, plus stacked per-channel
    amplitude and phase residual arrays (one row per pair, NaN at
    invalid AND clipped channels).
    """
    pair_rows = []
    amplitude_residuals = []
    phase_residuals = []

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

        bp1 = first["bandpass"]
        bp2 = second["bandpass"]
        flag1 = first["flag"]
        flag2 = second["flag"]

        if bp1.shape != bp2.shape:
            print(
                "  skipping pair because channel counts differ: "
                "{} versus {}".format(bp1.shape, bp2.shape)
            )
            continue

        valid = (
            (~flag1) & (~flag2)
            & np.isfinite(bp1) & np.isfinite(bp2)
            & (np.abs(bp1) > 0.0) & (np.abs(bp2) > 0.0)
        )

        amplitude = np.full(bp1.shape, np.nan, dtype=float)
        phase = np.full(bp1.shape, np.nan, dtype=float)

        ratio = bp2[valid] / bp1[valid]

        amplitude[valid] = np.log(np.abs(ratio))
        phase[valid] = np.angle(ratio)

        valid_channel_count = int(np.count_nonzero(valid))

        if valid_channel_count == 0:
            print("  skipping pair with no common valid channels")
            continue

        # Outlier rejection, independently for amplitude and phase,
        # restricted to already-valid channels.
        amplitude_keep = np.zeros(bp1.shape, dtype=bool)
        phase_keep = np.zeros(bp1.shape, dtype=bool)

        amplitude_keep[valid] = robust_sigma_clip_mask(
            amplitude[valid], clip_sigma=clip_sigma
        )
        phase_keep[valid] = robust_sigma_clip_mask(
            phase[valid], clip_sigma=clip_sigma
        )

        amplitude_clipped_count = int(np.count_nonzero(valid & ~amplitude_keep))
        phase_clipped_count = int(np.count_nonzero(valid & ~phase_keep))

        amplitude_rms = float(np.sqrt(np.nanmean(amplitude[amplitude_keep] ** 2)))
        phase_rms_radians = float(np.sqrt(np.nanmean(phase[phase_keep] ** 2)))

        # NaN out clipped channels in the arrays used for the
        # per-channel summary plot too, so a spike doesn't leak into
        # the percentile bands there either.
        amplitude_for_plot = amplitude.copy()
        amplitude_for_plot[valid & ~amplitude_keep] = np.nan

        phase_for_plot = phase.copy()
        phase_for_plot[valid & ~phase_keep] = np.nan

        pair_rows.append(
            {
                "pair_index": len(pair_rows),
                "first_table": first["bandpass_table"],
                "second_table": second["bandpass_table"],
                "first_antenna_index": first["antenna_index"],
                "second_antenna_index": second["antenna_index"],
                "first_time": first["time"],
                "second_time": second["time"],
                "dt_seconds": dt_seconds,
                "valid_channels": valid_channel_count,
                "amplitude_channels_clipped": amplitude_clipped_count,
                "phase_channels_clipped": phase_clipped_count,
                "amplitude_rms_log": amplitude_rms,
                "amplitude_rms_percent_approx": 100.0 * amplitude_rms,
                "phase_rms_degrees": float(np.degrees(phase_rms_radians)),
            }
        )

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

def write_pair_metrics(path, pair_rows):
    """
    Write one CSV row per adjacent table pair.
    """
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PAIR_METRICS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(pair_rows)


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
    Plot one amplitude and phase RMS value per adjacent pair, against
    real elapsed time (the midpoint of each pair's two epochs,
    relative to `reference_time` — the antenna's first observation).
    """
    import matplotlib.pyplot as plt

    first_times = np.array([row["first_time"] for row in pair_rows])
    second_times = np.array([row["second_time"] for row in pair_rows])

    # Get the mid-point
    representative_times = 0.5 * (first_times + second_times)
    relative_seconds = representative_times - reference_time

    time_values, time_unit = _relative_time_axis(relative_seconds)

    amplitude_rms_percent = np.array(
        [row["amplitude_rms_percent_approx"] for row in pair_rows]
    )
    phase_rms_degrees = np.array(
        [row["phase_rms_degrees"] for row in pair_rows]
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(time_values, amplitude_rms_percent, marker="o", markerfacecolor='skyblue',
        markeredgecolor='black',
        color='skyblue')
    axes[0].set_ylabel("Amplitude RMS [%]")
    axes[0].set_title("{}: adjacent-table amplitude residual".format(antenna_name))

    axes[1].plot(time_values, phase_rms_degrees, marker="o", markerfacecolor='hotpink',
        markeredgecolor='black',
        color='hotpink')
    axes[1].set_xlabel("Time relative to first epoch [{}]".format(time_unit))
    axes[1].set_ylabel("Phase RMS [deg]")
    axes[1].set_title("{}: adjacent-table phase residual".format(antenna_name))

    for axis in axes:
        axis.grid(True, alpha=0.3)

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
#    axes[0].set_ylim((-2,5))

    axes[1].plot(channels, phase_median, color='hotpink')
    axes[1].fill_between(channels, phase_low, phase_high, alpha=0.3, color='hotpink')
    axes[1].axhline(0.0, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Phase residual [deg]")
    axes[1].tick_params(axis='both', direction='in', length=6, color='black')
#    axes[1].set_ylim((-1,1))

    for axis in axes:
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# Process one physical antenna
# ============================================================

def process_antenna(antenna_name, mapping_rows, output_directory, spw, pol,
                     max_pair_gap_seconds=None, make_plots=True):
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
                os.path.basename(mapping_row["bandpass_table"]),
                mapping_row["antenna_index"], mean_time
            )
        )

    if len(records) < 2:
        print("  fewer than two usable observations; skipping")
        return

    records.sort(key=lambda record: record["time"])

    pair_rows, amplitude_residuals, phase_residuals = calculate_adjacent_residuals(
        records, max_pair_gap_seconds=max_pair_gap_seconds
    )

    if len(pair_rows) == 0:
        print("  no valid adjacent pairs; skipping outputs")
        return

    antenna_directory = os.path.join(output_directory, safe_filename(antenna_name))
    os.makedirs(antenna_directory, exist_ok=True)

    metrics_path = os.path.join(antenna_directory, f"{antenna_name}_pair_metrics.csv")
    write_pair_metrics(metrics_path, pair_rows)

    if make_plots:
        pair_plot_path = os.path.join(antenna_directory, f"{antenna_name}_pair_rms.png")
        channel_plot_path = os.path.join(antenna_directory, f"{antenna_name}_channel_summary.png")

        plot_pair_metrics(pair_plot_path, antenna_name, pair_rows, records[0]["time"])
        plot_channel_summary(
            channel_plot_path, antenna_name, amplitude_residuals, phase_residuals
        )

    print("  adjacent pairs:", len(pair_rows))
    print(
        "  median amplitude RMS [% approx.]: {:.6f}".format(
            np.nanmedian(
                [row["amplitude_rms_percent_approx"] for row in pair_rows]
            )
        )
    )
    print(
        "  median phase RMS [deg]: {:.6f}".format(
            np.nanmedian([row["phase_rms_degrees"] for row in pair_rows])
        )
    )
    print("  wrote:", antenna_directory)


# ============================================================
# Orchestration (called by the CLI)
# ============================================================

def run_lag_analysis(mapping_csv, output_directory="lag_residual_results",
                      spw=0, pol=0, only_antenna=None,
                      max_pair_gap_seconds=None, make_plots=True):
    """
    Run the full adjacent-pair lag analysis for every physical
    antenna present in `mapping_csv` at the given SPW.
    """
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
            max_pair_gap_seconds=max_pair_gap_seconds, make_plots=make_plots
        )

    print()
    print("Finished. Results are in:", output_directory)

    return output_directory
