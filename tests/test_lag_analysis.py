import numpy as np

import warnings

from bandpass_stability.lag_analysis import (
    average_bandpass_rows,
    calculate_adjacent_residuals,
    pair_metrics,
    time_direction_clip,
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


def _record(name, time, bandpass, flag=None):
    if flag is None:
        flag = np.zeros(bandpass.shape, dtype=bool)
    return {"bandpass_table": name, "antenna_index": 0, "time": time,
            "bandpass": bandpass, "flag": flag}


def test_average_bandpass_rows_is_silent_for_fully_flagged_channels():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        average_bandpass_rows([np.array([1.0 + 0j])], [np.array([True])], [0.0])


def test_phase_residual_near_180_degrees_is_not_split_by_wrapping():
    rng = np.random.default_rng(5)
    nchan = 500
    first = np.ones(nchan, dtype=complex)
    second = np.exp(1j * np.radians(179.0 + rng.normal(0, 2.0, nchan)))

    pair_rows, _, _ = calculate_adjacent_residuals(
        [_record("a", 0.0, first), _record("b", 10.0, second)]
    )

    assert pair_rows[0]["phase_channels_clipped"] < 5
    assert np.isclose(abs(pair_rows[0]["phase_mean_degrees"]), 179.0, atol=1.0)
    assert pair_rows[0]["phase_std_degrees"] < 3.0


def test_pair_metrics_separate_offset_scatter_and_delay():
    nchan = 1000
    channel = np.arange(nchan)
    width_hz = 200e3
    delay_s = 2e-9
    phase = 2 * np.pi * width_hz * delay_s * channel  # wraps a little across the band
    centred = np.angle(np.exp(1j * (phase - phase.mean())))
    keep = np.ones(nchan, dtype=bool)
    log_amp = np.full(nchan, np.log(1.02))

    metrics = pair_metrics(log_amp, centred, 0.0, keep, keep, chan_width_hz=width_hz)

    assert np.isclose(metrics["delay_ns"], 2.0, rtol=1e-2)
    assert metrics["phase_delay_removed_rms_degrees"] < 0.1
    assert np.isclose(metrics["amplitude_mean_log"], np.log(1.02))
    assert metrics["amplitude_std_log"] < 1e-12


def test_time_direction_clip_flags_wide_block_but_keeps_broadband_change():
    rng = np.random.default_rng(6)
    nchan = 800
    records = []
    for epoch in range(6):
        amplitude = np.exp(rng.normal(0, 0.01, nchan))
        phase = np.radians(rng.normal(0, 0.5, nchan))
        if epoch == 2:
            phase[100:140] += np.pi  # 40-channel half-turn block
        if epoch == 4:
            amplitude[400:] *= 1.03  # genuine broadband step
        records.append(_record(str(epoch), float(epoch), amplitude * np.exp(1j * phase)))

    counts = time_direction_clip(records, clip_sigma=5.0, window=380)

    assert records[2]["flag"][100:140].all()
    assert counts[2] >= 40
    assert counts[4] < 10  # the 3% step over half the band is kept
    assert not records[4]["flag"][450:].any()


def test_time_direction_clip_needs_three_epochs():
    records = [_record("a", 0.0, np.ones(50, dtype=complex)),
               _record("b", 1.0, -np.ones(50, dtype=complex))]
    assert time_direction_clip(records) == [0, 0]


def test_channels_clipped_in_most_pairs_are_masked_in_all():
    rng = np.random.default_rng(7)
    nchan = 600
    records = []
    for epoch in range(6):
        amplitude = np.exp(rng.normal(0, 0.01, nchan))
        # Channels 300-302 are contaminated in every epoch but one.
        if epoch != 3:
            amplitude[300:303] *= np.exp(rng.normal(0, 0.5, 3))
        records.append(_record(str(epoch), float(epoch), amplitude.astype(complex)))

    pair_rows, amplitude, _ = calculate_adjacent_residuals(records, mask_fraction=0.25)

    assert np.all(np.isnan(amplitude[:, 300:303]))
    assert all(row["amplitude_channels_used"] <= nchan - 3 for row in pair_rows)
