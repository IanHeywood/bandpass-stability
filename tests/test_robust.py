import numpy as np

from bandpass_stability.robust import (
    circular_centre,
    estimate_linear_phase,
    find_valid_segments,
    grow_flags,
    iterative_local_clip,
    iterative_stack_clip,
    masked_running_median,
    wrap_phase,
)


def test_find_valid_segments_handles_edges():
    valid = np.array([True, True, False, True])
    assert find_valid_segments(valid) == [(0, 2), (3, 4)]
    assert find_valid_segments(np.array([], dtype=bool)) == []


def test_grow_flags_widens_each_flagged_region():
    valid = np.array([True] * 5 + [False] * 2 + [True] * 5)
    grown = grow_flags(valid, 2)
    expected = np.array([True] * 3 + [False] * 6 + [True] * 3)
    np.testing.assert_array_equal(grown, expected)


def test_grow_flags_zero_is_a_no_op():
    valid = np.array([True, False, True])
    np.testing.assert_array_equal(grow_flags(valid, 0), valid)


def test_running_median_removes_feature_at_segment_edge():
    # With edge-copy padding ("nearest") a block at a segment edge wins
    # the median and survives; with reflection it is removed.
    values = np.zeros(200)
    values[:10] = 50.0
    valid = np.ones(200, dtype=bool)

    smoothed = masked_running_median(values, valid, valid, window=95)

    assert np.all(np.abs(smoothed[:10]) < 1e-9)


def test_running_median_bridges_unused_channels_within_a_segment():
    values = np.arange(20, dtype=float)
    valid = np.ones(20, dtype=bool)
    use = valid.copy()
    use[10] = False

    smoothed = masked_running_median(values, valid, use, window=3)

    assert np.isnan(smoothed[10])
    # Channel 9's window is {8, 9, 11}: the gap at 10 is bridged.
    assert smoothed[9] == 9.0
    assert smoothed[11] == 11.0


def test_wrap_phase_range():
    wrapped = wrap_phase(np.array([3 * np.pi, -3 * np.pi + 0.1, 0.5]))
    assert np.all(wrapped <= np.pi) and np.all(wrapped > -np.pi)
    assert np.isclose(wrapped[2], 0.5)


def test_circular_centre_near_pi_is_not_split():
    phases = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02, -np.pi + 0.02])
    centre = circular_centre(phases)
    assert np.isclose(abs(centre), np.pi, atol=0.02)


def test_estimate_linear_phase_recovers_wrapping_slope():
    channel = np.arange(1000)
    true_slope = 2 * np.pi * 3.0 / 1000  # three full turns across the band
    phase = wrap_phase(0.4 + true_slope * channel)
    use = np.ones(1000, dtype=bool)

    offset, slope = estimate_linear_phase(phase, use)

    assert np.isclose(slope, true_slope, rtol=1e-3)
    assert np.isclose(offset, 0.4, atol=1e-2)


def test_local_clip_keeps_smooth_structure_and_removes_spike():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 2000)
    values = 0.05 * x + 0.01 * np.sin(8 * np.pi * x) + rng.normal(0, 0.002, 2000)
    values[700:705] += 0.05
    valid = np.ones(2000, dtype=bool)

    keep = iterative_local_clip(values, valid, window=95, clip_sigma=5.0)

    assert not keep[700:705].any()
    assert np.count_nonzero(~keep) <= 10


def test_local_clip_catches_block_at_segment_edge():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 300)
    values[:15] += 12.0
    valid = np.ones(300, dtype=bool)

    keep = iterative_local_clip(values, valid, window=95, clip_sigma=5.0)

    assert not keep[:15].any()


def test_local_clip_wrap_ignores_two_pi_steps():
    rng = np.random.default_rng(2)
    phase = rng.normal(0, 0.01, 400)
    phase[200:] += 2 * np.pi  # an unwrapping artefact, not a real step
    valid = np.ones(400, dtype=bool)

    keep = iterative_local_clip(phase, valid, window=95, clip_sigma=5.0, wrap=True)

    assert keep.all()


def test_stack_clip_flags_wide_block_in_one_row_only():
    rng = np.random.default_rng(3)
    values = rng.normal(0, 1, (6, 600))
    values[2, 100:140] += 15.0
    use = np.ones(values.shape, dtype=bool)

    keep = iterative_stack_clip(values, use, window=380, clip_sigma=5.0)

    assert not keep[2, 100:140].any()
    assert np.count_nonzero(~keep) < 50
