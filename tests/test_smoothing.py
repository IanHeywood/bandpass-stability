import numpy as np

from bandpass_stability.smoothing import (
    choose_window,
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
