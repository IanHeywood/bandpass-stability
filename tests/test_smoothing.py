import numpy as np

from bandpass_stability.smoothing import (
    choose_window,
    clean_and_smooth_bandpass,
    find_valid_segments,
    smooth_complex_bandpass,
    smooth_real_array,
)


def test_find_valid_segments_basic():
    valid = np.array([False, True, True, False, True, False, False, True])
    assert find_valid_segments(valid) == [(1, 3), (4, 5), (7, 8)]


def test_find_valid_segments_all_valid():
    valid = np.array([True, True, True])
    assert find_valid_segments(valid) == [(0, 3)]


def test_find_valid_segments_all_flagged():
    valid = np.array([False, False, False])
    assert find_valid_segments(valid) == []


def test_choose_window_shrinks_to_odd_segment_length():
    # window larger than the segment: clip to segment length, then
    # nudge down to odd if needed.
    assert choose_window(95, 10) == 9
    assert choose_window(95, 11) == 11
    assert choose_window(3, 100) == 3


def test_choose_window_never_returns_less_than_one():
    assert choose_window(95, 0) == 1
    assert choose_window(95, 1) == 1


def test_smooth_real_array_removes_spike_within_window():
    valid = np.ones(21, dtype=bool)
    values = np.zeros(21)
    values[10] = 100.0  # single-channel spike

    smoothed = smooth_real_array(values, valid, window=5)

    assert smoothed[10] < 10.0  # spike is suppressed
    assert np.allclose(smoothed[:5], 0.0)  # untouched flat region


def test_smooth_real_array_never_crosses_a_flagged_gap():
    valid = np.array([True] * 5 + [False] * 3 + [True] * 5)
    values = np.concatenate([np.zeros(5), np.zeros(3), np.full(5, 10.0)])

    smoothed = smooth_real_array(values, valid, window=5)

    # The two valid segments have very different levels; smoothing
    # within each segment should not blend them together.
    assert np.allclose(smoothed[:5], 0.0)
    assert np.allclose(smoothed[13:], 10.0)
    assert np.all(np.isnan(smoothed[5:8]))  # gap stays NaN


def test_smooth_complex_bandpass_preserves_flagged_channels():
    nchan = 21
    rng = np.random.default_rng(1)

    amplitude = np.ones(nchan) + 0.01 * rng.normal(size=nchan)
    phase = 0.01 * rng.normal(size=nchan)
    spectrum = amplitude * np.exp(1j * phase)

    flags = np.zeros(nchan, dtype=bool)
    flags[10] = True
    original_flagged_value = spectrum[10]

    smoothed = smooth_complex_bandpass(spectrum, flags, window=5)

    # Flagged channels are left completely untouched.
    assert smoothed[10] == original_flagged_value


def test_smooth_complex_bandpass_reduces_amplitude_noise():
    nchan = 101
    rng = np.random.default_rng(2)

    true_amplitude = 1.0
    noisy_amplitude = true_amplitude + 0.2 * rng.normal(size=nchan)
    spectrum = noisy_amplitude.astype(complex)

    flags = np.zeros(nchan, dtype=bool)

    smoothed = smooth_complex_bandpass(spectrum, flags, window=21)

    noise_before = np.std(np.abs(spectrum) - true_amplitude)
    noise_after = np.std(np.abs(smoothed) - true_amplitude)

    assert noise_after < noise_before


def _noisy_bandpass(nchan=600, seed=4):
    rng = np.random.default_rng(seed)
    amplitude = 1.0 + 0.1 * np.sin(np.arange(nchan) / 80.0)
    amplitude = amplitude * np.exp(rng.normal(0, 0.01, nchan))
    phase = np.radians(rng.normal(0, 0.5, nchan))
    return amplitude * np.exp(1j * phase)


def test_smoothing_removes_spike_at_segment_edge():
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    flags[300:320] = True
    spectrum[298:300] *= 10.0  # spike right against the flagged gap

    smoothed = smooth_complex_bandpass(spectrum, flags, window=95)

    assert np.all(np.abs(smoothed[298:300]) < 2.0)


def test_clean_and_smooth_flags_edge_block_when_clipping():
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    flags[300:320] = True
    spectrum[285:300] *= np.exp(1j * np.radians(-12.0))

    output, output_flags, n_grown, n_clipped = clean_and_smooth_bandpass(
        spectrum, flags, window=95, clip_sigma=5.0
    )

    assert output_flags[285:300].all()
    assert n_clipped >= 15
    assert n_grown == 0
    # Original flags pass through; clipped channels keep their input values.
    assert output_flags[300:320].all()
    np.testing.assert_array_equal(output[285:300], spectrum[285:300])


def test_clean_and_smooth_without_options_leaves_flags_unchanged():
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    flags[100:110] = True

    _, output_flags, n_grown, n_clipped = clean_and_smooth_bandpass(spectrum, flags, window=95)

    np.testing.assert_array_equal(output_flags, flags)
    assert n_grown == 0 and n_clipped == 0


def test_clean_and_smooth_flag_growth():
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    flags[300:320] = True

    _, output_flags, n_grown, _ = clean_and_smooth_bandpass(
        spectrum, flags, window=95, flag_grow=3
    )

    assert n_grown == 6
    assert output_flags[297:323].all()
    assert not output_flags[296] and not output_flags[323]


def test_clean_and_smooth_handles_half_turn_block():
    # A 180-degree block can trip np.unwrap into a 2*pi step; the
    # clipping must still remove it and leave the rest untouched.
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    spectrum[400:405] *= -1.0

    output, output_flags, _, n_clipped = clean_and_smooth_bandpass(
        spectrum, flags, window=95, clip_sigma=5.0
    )

    assert output_flags[400:405].all()
    assert n_clipped <= 10
    kept = ~output_flags
    assert np.all(np.abs(np.degrees(np.angle(output[kept]))) < 5.0)


def test_smoothing_follows_a_steep_delay_slope():
    # Several full turns across the band must survive smoothing, even
    # though the phase is never unwrapped.
    rng = np.random.default_rng(8)
    nchan = 2000
    channel = np.arange(nchan)
    true_phase = 2 * np.pi * 5.0 * channel / nchan
    spectrum = np.exp(1j * (true_phase + np.radians(rng.normal(0, 1.0, nchan))))
    flags = np.zeros(nchan, dtype=bool)
    flags[900:950] = True

    smoothed = smooth_complex_bandpass(spectrum, flags, window=95)

    error = np.degrees(np.angle(smoothed[~flags] * np.exp(-1j * true_phase[~flags])))
    assert np.max(np.abs(error)) < 2.0


def test_smoothing_alone_is_not_disturbed_by_half_turn_block():
    spectrum = _noisy_bandpass()
    flags = np.zeros(spectrum.size, dtype=bool)
    spectrum[400:405] *= -1.0

    smoothed = smooth_complex_bandpass(spectrum, flags, window=95)

    # Without clipping the block stays unflagged, but the median
    # removes it and the rest of the band is unaffected.
    assert np.all(np.abs(np.degrees(np.angle(smoothed))) < 5.0)
