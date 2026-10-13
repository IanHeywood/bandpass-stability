"""
Command-line entry points.

These are deliberately thin: each one parses arguments and calls a
single orchestration function in `smoothing.py`, `antenna_mapping.py`,
or `lag_analysis.py`. The parsing/plumbing lives here; the actual
logic lives in those modules and can be reused without the CLI.
"""

import argparse

from .antenna_mapping import build_mapping
from .lag_analysis import (
    DEFAULT_CLIP_SIGMA,
    DEFAULT_CLIP_WINDOW,
    DEFAULT_MASK_FRACTION,
    DEFAULT_TIME_CLIP_SIGMA,
    TIME_CLIP_WINDOW_FACTOR,
    run_lag_analysis,
)
from .smoothing import smooth_tables


def smooth_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bandpass-smooth",
        description="Median-filter bandpass tables along the frequency axis.",
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern for input bandpass tables, e.g. 'cal_*.B0'.",
    )
    parser.add_argument(
        "--output-tag", default="_smooth",
        help="Suffix inserted before .B0 in output table names (default: _smooth).",
    )
    parser.add_argument(
        "--window", type=int, default=95,
        help="Median filter window size in channels; must be odd (default: 95).",
    )
    parser.add_argument(
        "--clip-sigma", type=float, default=None,
        help=(
            "Before smoothing, flag channels whose log-amplitude or phase "
            "deviates from a running-median baseline by more than this many "
            "local robust sigmas, iterating until nothing new is flagged. "
            "Writes to the FLAG column of the output table. "
            "Default: off. 5 is a reasonable starting point."
        ),
    )
    parser.add_argument(
        "--flag-grow", type=int, default=0,
        help=(
            "Before smoothing, extend every flagged region by this many "
            "channels on each side, to catch residual RFI in the wings of "
            "already-flagged regions. Writes to the FLAG column of the "
            "output table. Default: 0 (off)."
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip writing diagnostic PNGs.",
    )

    args = parser.parse_args(argv)

    smooth_tables(
        input_pattern=args.pattern,
        output_tag=args.output_tag,
        window=args.window,
        make_plots=not args.no_plots,
        clip_sigma=args.clip_sigma,
        flag_grow=args.flag_grow,
    )


def map_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bandpass-map",
        description=(
            "Build a CSV mapping each bandpass-table antenna index to "
            "its physical antenna name in the parent Measurement Set."
        ),
    )
    parser.add_argument(
        "pattern",
        help=(
            "Glob pattern for input bandpass tables, e.g. "
            "'cal_*_smooth.B0'. Be specific here: a pattern that also "
            "matches un-smoothed tables will double-count observations."
        ),
    )
    parser.add_argument(
        "-o", "--output-csv", default="bandpass_antenna_mapping.csv",
        help="Output CSV path (default: bandpass_antenna_mapping.csv).",
    )

    args = parser.parse_args(argv)

    build_mapping(
        bandpass_pattern=args.pattern,
        output_csv=args.output_csv,
    )


def lag_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="bandpass-lag",
        description=(
            "Compute adjacent-epoch bandpass stability metrics per "
            "physical antenna from a mapping CSV."
        ),
    )
    parser.add_argument(
        "mapping_csv",
        help="Mapping CSV produced by 'bandpass-map'.",
    )
    parser.add_argument(
        "-o", "--output-directory", default="lag_residual_results",
        help="Output directory (default: lag_residual_results).",
    )
    parser.add_argument(
        "--spw", type=int, default=0,
        help="Spectral window ID to analyse (default: 0).",
    )
    parser.add_argument(
        "--pol", type=int, default=0,
        help="Polarization index to analyse (default: 0).",
    )
    parser.add_argument(
        "--only-antenna", default=None,
        help="Restrict to a single antenna name (default: all antennas).",
    )
    parser.add_argument(
        "--max-pair-gap-seconds", type=float, default=None,
        help="Skip adjacent pairs separated by more than this many seconds.",
    )
    parser.add_argument(
        "--clip-sigma", type=float, default=DEFAULT_CLIP_SIGMA,
        help=(
            "Per-pair outlier threshold, in local robust sigmas about a "
            "running-median baseline along frequency. 0 disables "
            "(default: {}).".format(DEFAULT_CLIP_SIGMA)
        ),
    )
    parser.add_argument(
        "--clip-window", type=int, default=DEFAULT_CLIP_WINDOW,
        help=(
            "Running-median window in channels for the per-pair clip; "
            "must be odd. The time-direction clip uses {}x this "
            "(default: {}).".format(TIME_CLIP_WINDOW_FACTOR, DEFAULT_CLIP_WINDOW)
        ),
    )
    parser.add_argument(
        "--time-clip-sigma", type=float, default=DEFAULT_TIME_CLIP_SIGMA,
        help=(
            "Threshold for flagging channels in individual epochs that "
            "deviate from the other epochs locally in frequency. Needs at "
            "least three epochs. 0 disables (default: {}).".format(
                DEFAULT_TIME_CLIP_SIGMA)
        ),
    )
    parser.add_argument(
        "--mask-fraction", type=float, default=DEFAULT_MASK_FRACTION,
        help=(
            "Exclude from every pair any channel clipped in more than this "
            "fraction of pairs, so pairs share a channel set. 1 disables "
            "(default: {}).".format(DEFAULT_MASK_FRACTION)
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip writing diagnostic PNGs.",
    )

    args = parser.parse_args(argv)

    run_lag_analysis(
        mapping_csv=args.mapping_csv,
        output_directory=args.output_directory,
        spw=args.spw,
        pol=args.pol,
        only_antenna=args.only_antenna,
        max_pair_gap_seconds=args.max_pair_gap_seconds,
        make_plots=not args.no_plots,
        clip_sigma=args.clip_sigma,
        clip_window=args.clip_window,
        time_clip_sigma=args.time_clip_sigma,
        mask_fraction=args.mask_fraction,
    )
