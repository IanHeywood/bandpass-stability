import numpy as np

from bandpass_stability.lag_analysis import (
    average_bandpass_rows,
    calculate_adjacent_residuals,
)


def test_average_bandpass_rows_ignores_flagged_samples():
    # Two rows; channel 0 has one flagged sample that should be
    # excluded from the average.
    spectra = [
        np.array([1.0 + 0j, 2.0 + 0j]),
        np.array([9.0 + 0j, 4.0 + 0j]),  # channel 0 here is flagged
    ]
    flags = [
        np.array([False, False]),
        np.array([True, False]),
    ]
    times = [100.0, 200.0]

    mean_time, averaged, averaged_flag = average_bandpass_rows(spectra, flags, times)

    assert mean_time == 150.0
    np.testing.assert_allclose(averaged[0], 1.0)  # only the unflagged sample
    np.testing.assert_allclose(averaged[1], 3.0)  # mean of 2 and 4
    assert not averaged_flag[0]
    assert not averaged_flag[1]


def test_average_bandpass_rows_flags_channel_with_no_valid_samples():
    spectra = [np.array([0.0 + 0j]), np.array([0.0 + 0j])]
    flags = [np.array([True]), np.array([True])]
    times = [0.0, 10.0]

    _, averaged, averaged_flag = average_bandpass_rows(spectra, flags, times)

    assert averaged_flag[0]
    assert np.isnan(averaged[0])


def test_calculate_adjacent_residuals_zero_for_identical_epochs():
    bandpass = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
    flag = np.zeros(3, dtype=bool)

    records = [
        {"bandpass_table": "a", "antenna_index": 3, "time": 0.0,
         "bandpass": bandpass.copy(), "flag": flag},
        {"bandpass_table": "b", "antenna_index": 5, "time": 10.0,
         "bandpass": bandpass.copy(), "flag": flag},
    ]

    pair_rows, amplitude, phase = calculate_adjacent_residuals(records)

    assert len(pair_rows) == 1
    assert pair_rows[0]["amplitude_rms_percent_approx"] == 0.0
    assert pair_rows[0]["phase_rms_degrees"] == 0.0
    # Antenna index is allowed to differ between epochs; it is
    # tracked per-record, not assumed constant.
    assert pair_rows[0]["first_antenna_index"] == 3
    assert pair_rows[0]["second_antenna_index"] == 5


def test_calculate_adjacent_residuals_detects_amplitude_change():
    flag = np.zeros(2, dtype=bool)

    records = [
        {"bandpass_table": "a", "antenna_index": 0, "time": 0.0,
         "bandpass": np.array([1.0 + 0j, 1.0 + 0j]), "flag": flag},
        {"bandpass_table": "b", "antenna_index": 0, "time": 10.0,
         "bandpass": np.array([2.0 + 0j, 2.0 + 0j]), "flag": flag},
    ]

    pair_rows, _, _ = calculate_adjacent_residuals(records)

    expected_log_ratio = np.log(2.0)
    assert np.isclose(pair_rows[0]["amplitude_rms_log"], expected_log_ratio)


def test_calculate_adjacent_residuals_respects_max_gap():
    flag = np.zeros(2, dtype=bool)
    bandpass = np.array([1.0 + 0j, 1.0 + 0j])

    records = [
        {"bandpass_table": "a", "antenna_index": 0, "time": 0.0,
         "bandpass": bandpass.copy(), "flag": flag},
        {"bandpass_table": "b", "antenna_index": 0, "time": 1000.0,
         "bandpass": bandpass.copy(), "flag": flag},
    ]

    pair_rows, _, _ = calculate_adjacent_residuals(
        records, max_pair_gap_seconds=100.0
    )

    assert pair_rows == []


def test_calculate_adjacent_residuals_skips_channels_flagged_in_either_epoch():
    flag_first = np.array([False, True])
    flag_second = np.array([False, False])

    records = [
        {"bandpass_table": "a", "antenna_index": 0, "time": 0.0,
         "bandpass": np.array([1.0 + 0j, 1.0 + 0j]), "flag": flag_first},
        {"bandpass_table": "b", "antenna_index": 0, "time": 10.0,
         "bandpass": np.array([1.0 + 0j, 5.0 + 0j]), "flag": flag_second},
    ]

    pair_rows, amplitude, phase = calculate_adjacent_residuals(records)

    assert pair_rows[0]["valid_channels"] == 1
    assert np.isnan(amplitude[0, 1])  # channel flagged in first epoch is excluded
