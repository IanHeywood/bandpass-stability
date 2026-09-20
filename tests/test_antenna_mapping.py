import os

import pytest

from bandpass_stability.antenna_mapping import parent_ms_from_bandpass_name


def test_parent_ms_from_bandpass_name_strips_cal_prefix(tmp_path):
    ms_path = tmp_path / "observation.ms"
    ms_path.mkdir()

    bandpass_path = tmp_path / "cal_observation.ms.B0"
    bandpass_path.mkdir()

    resolved = parent_ms_from_bandpass_name(str(bandpass_path))

    assert resolved == str(ms_path)


def test_parent_ms_from_bandpass_name_works_on_smoothed_tables(tmp_path):
    # This is the exact case that matters for pipeline wiring: the
    # smoothed-table name still resolves to the same parent MS as the
    # raw table it was derived from.
    ms_path = tmp_path / "observation.ms"
    ms_path.mkdir()

    raw_bandpass = tmp_path / "cal_observation.ms.B0"
    raw_bandpass.mkdir()

    smoothed_bandpass = tmp_path / "cal_observation.ms_smooth.B0"
    smoothed_bandpass.mkdir()

    assert parent_ms_from_bandpass_name(str(raw_bandpass)) == str(ms_path)
    assert parent_ms_from_bandpass_name(str(smoothed_bandpass)) == str(ms_path)


def test_parent_ms_from_bandpass_name_without_cal_prefix(tmp_path):
    ms_path = tmp_path / "observation.ms"
    ms_path.mkdir()

    bandpass_path = tmp_path / "observation.ms.B0"
    bandpass_path.mkdir()

    resolved = parent_ms_from_bandpass_name(str(bandpass_path))

    assert resolved == str(ms_path)


def test_parent_ms_from_bandpass_name_missing_marker_raises(tmp_path):
    bandpass_path = tmp_path / "cal_observation.B0"
    bandpass_path.mkdir()

    with pytest.raises(RuntimeError, match="does not contain"):
        parent_ms_from_bandpass_name(str(bandpass_path))


def test_parent_ms_from_bandpass_name_missing_ms_raises(tmp_path):
    bandpass_path = tmp_path / "cal_observation.ms.B0"
    bandpass_path.mkdir()
    # Deliberately do not create observation.ms

    with pytest.raises(RuntimeError, match="does not exist"):
        parent_ms_from_bandpass_name(str(bandpass_path))
