"""
Command-line entry points.

These are deliberately thin: each one parses arguments and calls a
single orchestration function in `smoothing.py`, `antenna_mapping.py`,
or `lag_analysis.py`. The parsing/plumbing lives here; the actual
logic lives in those modules and can be reused without the CLI.
"""

import argparse

from .antenna_mapping import build_mapping
from .lag_analysis import run_lag_analysis
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
        "--no-plots", action="store_true",
        help="Skip writing diagnostic PNGs.",
    )

    args = parser.parse_args(argv)

    smooth_tables(
        input_pattern=args.pattern,
        output_tag=args.output_tag,
        window=args.window,
        make_plots=not args.no_plots,
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
    )
