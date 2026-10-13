"""
Robust statistics shared by the smoothing and lag-analysis stages.

Everything here works on plain numpy arrays, so it can be unit-tested
without casacore.

The central idea is clipping against a *local* baseline: a running
median along frequency, computed only from channels currently
believed to be good, with a running (local) MAD as the noise scale.
Smooth genuine structure (band edges, ripple, slopes) is followed by
the baseline and so is not clipped, while features that are narrow
compared with the window stand out. Heteroscedastic noise (noisier
band edges, channels next to RFI) is handled by the local MAD.
"""

import warnings

import numpy as np
from scipy.ndimage import median_filter

# Scale factor so a MAD approximates the standard deviation of a
# Gaussian core distribution.
MAD_TO_SIGMA = 1.4826

# The local noise scale is never allowed to fall below this fraction
# of the global (whole-spectrum) robust sigma. This stops quiet
# regions, or plateaus produced by an earlier median filter, from
# yielding a near-zero scale that would flag ordinary noise.
LOCAL_SIGMA_FLOOR_FRACTION = 0.5

# Safety net for iterative loops that should converge on their own.
MAX_CLIP_ITERATIONS = 100


# ============================================================
# Masks and segments
# ============================================================

def find_valid_segments(valid):
    """
    Return contiguous True regions of a boolean array as
    (start, end) pairs. `end` is exclusive.
    """
    valid = np.asarray(valid, dtype=bool)

    if valid.size == 0:
        return []

    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))

    return [(int(start), int(end)) for start, end in zip(changes[0::2], changes[1::2])]


def choose_window(requested_window, segment_length):
    """
    Choose an odd window size no larger than the segment it will be
    applied to.
    """
    window = min(requested_window, segment_length)

    if window % 2 == 0:
        window -= 1

    return max(window, 1)


def grow_flags(valid, n_channels):
    """
    Shrink the valid mask by `n_channels` on either side of every
    invalid region, i.e. grow the flags into their wings.
    """
    valid = np.asarray(valid, dtype=bool)

    if n_channels <= 0 or valid.all() or not valid.any():
        return valid.copy()

    invalid = ~valid
    grown = invalid.copy()

    for shift in range(1, n_channels + 1):
        grown[shift:] |= invalid[:-shift]
        grown[:-shift] |= invalid[shift:]

    return ~grown


# ============================================================
# Running statistics
# ============================================================

def masked_running_median(values, segments_valid, use, window, mode="reflect"):
    """
    Running median of `values` along the last axis.

    Segments are defined by `segments_valid` (e.g. the original flag
    mask): the filter never crosses a gap between segments. Within a
    segment, only channels where `use` is True contribute, and they
    are filtered as one contiguous run, so channels removed by
    clipping are bridged rather than turned into new segment edges.

    `mode="reflect"` mirrors the data at segment edges. The previous
    `mode="nearest"` padded with copies of the edge channel, which
    lets any feature sitting at a segment edge win the median and
    survive untouched.

    Returns NaN wherever `use` is False.
    """
    values = np.asarray(values, dtype=float)
    use = np.asarray(use, dtype=bool) & np.asarray(segments_valid, dtype=bool)

    output = np.full(values.shape, np.nan, dtype=float)

    for start, end in find_valid_segments(segments_valid):
        index = np.arange(start, end)[use[start:end]]

        if index.size == 0:
            continue

        local_window = choose_window(window, index.size)

        output[index] = median_filter(values[index], size=local_window, mode=mode)

    return output


def robust_sigma(values):
    """
    MAD-based sigma of the finite entries of `values`. Returns NaN
    if fewer than three finite values are available.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 3:
        return np.nan

    median = np.median(values)

    return MAD_TO_SIGMA * float(np.median(np.abs(values - median)))


def local_robust_sigma(residual, segments_valid, use, window):
    """
    Local noise scale: a running median of |residual| (a running
    MAD, since the residual is already relative to a running
    median), floored at a fraction of the global robust sigma.

    Returns (local_sigma_array, global_sigma). The array is NaN where
    `use` is False.
    """
    residual = np.asarray(residual, dtype=float)
    use = np.asarray(use, dtype=bool)

    global_sigma = robust_sigma(residual[use])

    local = MAD_TO_SIGMA * masked_running_median(
        np.abs(residual), segments_valid, use, window
    )

    if np.isfinite(global_sigma):
        local = np.fmax(local, LOCAL_SIGMA_FLOOR_FRACTION * global_sigma)
        local[~use] = np.nan

    return local, global_sigma


# ============================================================
# Phase helpers
# ============================================================

def wrap_phase(phase):
    """
    Wrap phase(s) in radians into (-pi, pi].
    """
    return np.angle(np.exp(1j * np.asarray(phase, dtype=float)))


def circular_centre(phase, use=None):
    """
    Robust central value of a set of phases (radians): the angle of
    the component-wise median of the unit phasors. Safe for phases
    clustered around +/-pi, unlike a plain median.
    """
    phase = np.asarray(phase, dtype=float)

    if use is None:
        use = np.isfinite(phase)

    selected = phase[np.asarray(use, dtype=bool) & np.isfinite(phase)]

    if selected.size == 0:
        return 0.0

    return float(np.angle(np.median(np.cos(selected)) + 1j * np.median(np.sin(selected))))


def robust_linear_fit(x, y, use, clip_sigma=5.0, n_iterations=5):
    """
    Least-squares straight line y = a + b x over channels where `use`
    is True, refitted a few times with outliers removed.

    Returns (a, b). Falls back to (median, 0) with fewer than three
    points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.asarray(use, dtype=bool) & np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(keep) < 3:
        finite = y[keep]
        return (float(np.median(finite)) if finite.size else 0.0), 0.0

    a = b = 0.0

    for _ in range(n_iterations):
        design = np.vstack([np.ones(np.count_nonzero(keep)), x[keep]]).T
        (a, b), *_ = np.linalg.lstsq(design, y[keep], rcond=None)

        residual = y - (a + b * x)
        sigma = robust_sigma(residual[keep])

        if not np.isfinite(sigma) or sigma == 0:
            break

        new_keep = keep & (np.abs(residual) <= clip_sigma * sigma)

        if np.count_nonzero(new_keep) < 3 or np.array_equal(new_keep, keep):
            break

        keep = new_keep

    return float(a), float(b)


def estimate_linear_phase(phase, use, oversample=8):
    """
    Fit phase(channel) = offset + slope * channel to wrapped phases,
    robust to wraps across the band (i.e. large delays).

    A coarse slope is found from the peak of a zero-padded FFT of the
    unit phasors, then refined with a robust straight-line fit to the
    remaining (now small) wrapped residual.

    Returns (offset_radians, slope_radians_per_channel). Channel
    numbers are the array indices, so the slope converts to a delay
    via the channel width.
    """
    phase = np.asarray(phase, dtype=float)
    use = np.asarray(use, dtype=bool) & np.isfinite(phase)

    nchan = phase.size
    channel = np.arange(nchan, dtype=float)

    if np.count_nonzero(use) < 3:
        return circular_centre(phase, use), 0.0

    nfft = 1 << int(np.ceil(np.log2(max(oversample * nchan, 2))))

    phasors = np.zeros(nfft, dtype=complex)
    phasors[:nchan][use] = np.exp(1j * phase[use])

    spectrum = np.abs(np.fft.fft(phasors))
    peak = int(np.argmax(spectrum))

    cycles_per_channel = peak / nfft
    if cycles_per_channel >= 0.5:
        cycles_per_channel -= 1.0

    coarse_slope = 2.0 * np.pi * cycles_per_channel

    derotated = wrap_phase(phase - coarse_slope * channel)
    centre = circular_centre(derotated, use)
    derotated = wrap_phase(derotated - centre)

    a, b = robust_linear_fit(channel, derotated, use)

    return float(wrap_phase(centre + a)), float(coarse_slope + b)


# ============================================================
# Iterative clipping
# ============================================================

def iterative_local_clip(values, segments_valid, window, clip_sigma,
                         wrap=False, use=None):
    """
    Iteratively flag channels that deviate from a running-median
    baseline by more than `clip_sigma` times the local robust sigma.

    `values` may be 1D, or 2D with several quantities stacked along
    the first axis (e.g. log-amplitude and phase); a channel is
    rejected if *any* quantity is an outlier. `wrap` may be a bool or
    a sequence of bools, one per quantity, and wraps that quantity's
    residual into (-pi, pi] before testing.

    Iterates until a pass rejects no further channels.

    Returns a boolean mask, True for channels to KEEP.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    segments_valid = np.asarray(segments_valid, dtype=bool)

    if np.isscalar(wrap) or isinstance(wrap, bool):
        wrap = [bool(wrap)] * values.shape[0]

    keep = segments_valid & np.all(np.isfinite(values), axis=0)

    if use is not None:
        keep &= np.asarray(use, dtype=bool)

    if clip_sigma is None or clip_sigma <= 0:
        return keep

    for _ in range(MAX_CLIP_ITERATIONS):
        if np.count_nonzero(keep) < 3:
            break

        reject = np.zeros_like(keep)

        for quantity, wrap_this in zip(values, wrap):
            residual = residual_from_baseline(
                quantity, segments_valid, keep, window, wrap=wrap_this
            )
            sigma, global_sigma = local_robust_sigma(
                residual, segments_valid, keep, window
            )

            if not np.isfinite(global_sigma) or global_sigma == 0:
                continue

            with np.errstate(invalid="ignore"):
                reject |= keep & (np.abs(residual) > clip_sigma * sigma)

        if not reject.any():
            break

        keep = keep & ~reject

    return keep


def circular_running_median(phase, segments_valid, use, window):
    """
    Running median of phases (radians) that cannot be fooled by
    wrapping: no unwrap is ever performed.

    A coarse baseline is the angle of the running medians of cos and
    sin (the component-wise median of the unit phasors). It is used
    only to choose each channel's 2*pi branch (the one nearest the
    coarse baseline); an ordinary running median of those phases is
    then the result. Where nothing wraps this equals the median of the
    unwrapped phase, but a short half-turn block cannot capture the
    baseline, which it can when the phase is unwrapped first: np.unwrap
    may turn such a block into a 2*pi step in the rest of the segment,
    and the median at the step lands on the block.

    Assumes the phase changes by well under half a turn across one
    window; remove any large delay slope first (see
    `estimate_linear_phase`).

    Returns the baseline (not wrapped), NaN where `use` is False.
    """
    phase = np.asarray(phase, dtype=float)

    cos_median = masked_running_median(np.cos(phase), segments_valid, use, window)
    sin_median = masked_running_median(np.sin(phase), segments_valid, use, window)

    coarse = np.arctan2(sin_median, cos_median)

    branch_resolved = coarse + wrap_phase(phase - coarse)

    return masked_running_median(branch_resolved, segments_valid, use, window)


def residual_from_baseline(values, segments_valid, use, window, wrap=False):
    """
    `values` minus their running-median baseline (NaN where `use` is
    False). With `wrap=True` the values are phases in radians: the
    baseline is a `circular_running_median` and the residual is
    wrapped into (-pi, pi].
    """
    values = np.asarray(values, dtype=float)

    if wrap:
        baseline = circular_running_median(values, segments_valid, use, window)
    else:
        baseline = masked_running_median(values, segments_valid, use, window)

    residual = values - baseline

    if wrap:
        residual = wrap_phase(residual)
        residual[~np.isfinite(baseline)] = np.nan

    return residual


def quiet_nanmedian(values, axis=None):
    """
    np.nanmedian without the 'All-NaN slice' RuntimeWarning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


# ============================================================
# Clipping a stack of spectra (epochs or pairs) together
# ============================================================

# Minimum number of rows (epochs or pairs) for the per-channel noise
# scale to be estimated across the stack; with fewer, each row falls
# back to its own local robust sigma along frequency.
MIN_ROWS_FOR_STACK_SIGMA = 3


def stack_robust_sigma(residuals, keep, segments_valid, window):
    """
    Per-channel noise scale for a stack of residual spectra (rows are
    epochs or pairs): the MAD across rows at each channel, smoothed
    along frequency with a running median.

    Measuring the scatter across rows, channel by channel, follows
    noise that changes slowly across the band (e.g. smoothed spectra
    are noisier near segment edges, where the filter averages fewer
    channels) without mistaking that for outliers. Smoothing it along
    frequency (within each run of usable channels) stops a feature
    that recurs in the same few channels (narrower than about half the
    window) from inflating its own noise scale and escaping the clip.

    Returns an array broadcastable against `residuals`.
    """
    residuals = np.asarray(residuals, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    nrows, nchan = residuals.shape

    if nrows < MIN_ROWS_FOR_STACK_SIGMA:
        sigma = np.full(residuals.shape, np.nan)
        for row in range(nrows):
            sigma[row], _ = local_robust_sigma(residuals[row], segments_valid, keep[row], window)
        return sigma

    absolute = np.where(keep, np.abs(residuals), np.nan)
    rows_per_channel = np.sum(keep, axis=0)

    per_channel = MAD_TO_SIGMA * quiet_nanmedian(absolute, axis=0)
    per_channel[rows_per_channel < MIN_ROWS_FOR_STACK_SIGMA] = np.nan

    usable = np.isfinite(per_channel)
    if not usable.any():
        return np.full(nchan, np.nan)

    # Smooth within contiguous runs of usable channels (i.e. between
    # flagged gaps), with reflection at their edges, so that the
    # genuine rise in noise towards a gap edge is kept rather than
    # averaged away by channels on the far side of the gap.
    smoothed = masked_running_median(per_channel, usable, usable, window)

    # Fill channels without enough rows from their neighbours.
    channel = np.arange(nchan)
    smoothed = np.interp(channel, channel[usable], smoothed[usable])

    global_sigma = robust_sigma(residuals[keep])
    if np.isfinite(global_sigma):
        smoothed = np.fmax(smoothed, LOCAL_SIGMA_FLOOR_FRACTION * global_sigma)

    return smoothed


def iterative_stack_clip(values, use, window, clip_sigma, wrap=False,
                         segments_valid=None):
    """
    Clip a stack of spectra (rows: epochs or pairs; columns: channels)
    against a running-median baseline along frequency for each row,
    with the per-channel noise scale measured across rows (see
    `stack_robust_sigma`). Iterates until a pass rejects nothing new.

    `segments_valid` defaults to the whole band, appropriate for
    ratios and deviations that are roughly continuous across flagged
    gaps.

    Returns a boolean array like `values`, True for samples to KEEP.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    use = np.atleast_2d(np.asarray(use, dtype=bool))

    if segments_valid is None:
        segments_valid = np.ones(values.shape[1], dtype=bool)

    keep = use & np.isfinite(values)

    if clip_sigma is None or clip_sigma <= 0:
        return keep

    for _ in range(MAX_CLIP_ITERATIONS):
        residuals = np.vstack([
            residual_from_baseline(values[row], segments_valid, keep[row], window, wrap=wrap)
            for row in range(values.shape[0])
        ])

        sigma = stack_robust_sigma(residuals, keep, segments_valid, window)

        with np.errstate(invalid="ignore"):
            reject = keep & (np.abs(residuals) > clip_sigma * sigma)

        if not reject.any():
            break

        keep = keep & ~reject

    return keep
