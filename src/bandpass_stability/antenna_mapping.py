"""
Build a CSV mapping each bandpass-table antenna index to the physical
antenna name in its parent Measurement Set.

This exists because CASA bandpass tables record antenna solutions by
integer index only. The same index can refer to a different physical
antenna in different epochs or array builds, so every downstream 
comparison must track the antenna *name*, looked up pet table, rather 
than assuming the index is stable over the set.

It is assumed that the name of each bandpass table contains the complete 
parent MS name, e.g. for a bandpass table named `cal_observation.ms.B0`
the parent MS would be `observation.ms`.
"""

import csv
import glob
import os

import numpy as np

from .caltable_utils import channel_and_pol_count, natural_sort_key, normalise_name

FIELDNAMES = [
    "bandpass_table",
    "parent_ms",
    "mean_time",
    "antenna_index",
    "antenna_name",
    "spw",
    "scan_number",
    "solution_rows",
    "nchan",
    "npol",
]


# ============================================================
# File helpers
# ============================================================

def find_bandpass_tables(pattern):
    """
    Find and naturally sort the input bandpass tables.
    """
    paths = sorted(glob.glob(pattern), key=natural_sort_key)

    if len(paths) == 0:
        raise RuntimeError("No bandpass tables matched: {}".format(pattern))

    return paths


def parent_ms_from_bandpass_name(bandpass_path):
    """
    Derive the parent MS path from a bandpass-table name.

    Everything up to and including the first '.ms' is used; a
    leading 'cal_' prefix on the table's basename is stripped.
    """
    directory = os.path.dirname(os.path.abspath(bandpass_path))
    basename = os.path.basename(bandpass_path)

    marker = ".ms"

    if marker not in basename:
        raise RuntimeError(
            "Bandpass-table name does not contain '.ms': {}".format(bandpass_path)
        )

    end = basename.index(marker) + len(marker)
    ms_basename = basename[:end]

    if ms_basename.startswith("cal_"):
        ms_basename = ms_basename[4:]

    ms_path = os.path.join(directory, ms_basename)

    if not os.path.isdir(ms_path):
        raise RuntimeError(
            "Derived parent MS does not exist:\n"
            "  bandpass table: {}\n"
            "  expected MS:    {}".format(bandpass_path, ms_path)
        )

    return ms_path


# ============================================================
# Read metadata
# ============================================================

def read_antenna_names(ms_path):
    """
    Read physical antenna names from the MS ANTENNA subtable.

    The row number in the ANTENNA table is the antenna index.
    """
    from casacore.tables import table

    antenna_table_path = os.path.join(ms_path, "ANTENNA")

    ant_tb = table(antenna_table_path, readonly=True, ack=False)
    names = ant_tb.getcol("NAME")
    ant_tb.close()

    return [normalise_name(name) for name in names]


def inspect_bandpass_table(bandpass_path, ms_path):
    """
    Return one mapping row for every antenna/SPW combination in one
    bandpass table.
    """
    from casacore.tables import table

    antenna_names = read_antenna_names(ms_path)

    bp_tb = table(bandpass_path, readonly=True, ack=False)

    try:
        antenna_column = bp_tb.getcol("ANTENNA1")
        spw_column = bp_tb.getcol("SPECTRAL_WINDOW_ID")
        time_column = bp_tb.getcol("TIME")

        if "SCAN_NUMBER" in bp_tb.colnames():
            scan_column = bp_tb.getcol("SCAN_NUMBER")
        else:
            scan_column = None

        combinations = sorted(
            set(zip(antenna_column.tolist(), spw_column.tolist()))
        )

        output_rows = []

        for antenna_index, spw in combinations:
            matching_rows = np.where(
                (antenna_column == antenna_index) & (spw_column == spw)
            )[0]

            if antenna_index < 0 or antenna_index >= len(antenna_names):
                raise RuntimeError(
                    "Antenna index {} in {} is outside the ANTENNA table "
                    "for {}".format(antenna_index, bandpass_path, ms_path)
                )

            mean_time = float(np.mean(time_column[matching_rows]))

            if scan_column is None:
                scan_number = ""
            else:
                unique_scans = np.unique(scan_column[matching_rows])
                scan_number = ",".join(str(int(value)) for value in unique_scans)

            cparam = np.asarray(bp_tb.getcell("CPARAM", int(matching_rows[0])))
            nchan, npol = channel_and_pol_count(cparam.shape)

            output_rows.append(
                {
                    "bandpass_table": os.path.abspath(bandpass_path),
                    "parent_ms": os.path.abspath(ms_path),
                    "mean_time": mean_time,
                    "antenna_index": int(antenna_index),
                    "antenna_name": antenna_names[antenna_index],
                    "spw": int(spw),
                    "scan_number": scan_number,
                    "solution_rows": int(len(matching_rows)),
                    "nchan": int(nchan),
                    "npol": int(npol),
                }
            )
    finally:
        bp_tb.close()

    return output_rows


# ============================================================
# Orchestration (called by the CLI)
# ============================================================

def build_mapping(bandpass_pattern, output_csv="bandpass_antenna_mapping.csv"):
    """
    Build the antenna-index-to-name mapping CSV for every bandpass
    table matching `bandpass_pattern`.
    """
    bandpass_tables = find_bandpass_tables(bandpass_pattern)

    print("Found {} bandpass tables:".format(len(bandpass_tables)))

    all_rows = []

    for index, bandpass_path in enumerate(bandpass_tables):
        print()
        print("[{}/{}] {}".format(index + 1, len(bandpass_tables), bandpass_path))

        parent_ms = parent_ms_from_bandpass_name(bandpass_path)
        print("  parent MS:", parent_ms)

        mapping_rows = inspect_bandpass_table(bandpass_path, parent_ms)

        for row in mapping_rows:
            print(
                "  index {:2d} -> {:s}, SPW {}, rows {}".format(
                    row["antenna_index"], row["antenna_name"],
                    row["spw"], row["solution_rows"]
                )
            )

        all_rows.extend(mapping_rows)

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("Wrote {} mapping rows to {}".format(len(all_rows), output_csv))
    print()
    print("Please inspect the CSV before running the lag analysis.")

    return output_csv
