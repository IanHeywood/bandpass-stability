# bandpass-stability

The aim of these tools is to gauge the spectral and temporal stability of a radio interferometer. 
Please read this page prior to use, as there are some important caveats, particularly with respect
to reference antennas.

## Installation

Clone this repo, create a fresh python virtual environment and you can `pip install` it. 
Using `pip install -e` is recommended, as the project should be considered to be under
active development and as always some bugs will probably be uncovered as it gets used
on more results. The tests can be run with `pip install -e .[dev]` followed by `pytest`;
they do not need casacore.


## Method

The input product for these tools is a set of CASA-format bandpass tables. Support for 
other formats can be added as required. Each table will at present be treated as a single 
epoch, with stability metrics being computed by analysing the properties of the 
bandpass over the whole set. Intra-table processing for very short timescale stability 
analyses is also pending, and will likely be added as a separate command line utility.
Note that the input tables used for stability analyses should _not_ be normalised, as this
would completely obfuscate instrumental amplitude drift between epochs.

The workflow and each associated tool is as follows:

1. `bandpass-smooth`: A perfectly stable instrument will still exhibit 
    apparent temporal variations in its bandpass response due to the unavoidable
    effects of thermal noise. To minimise the contribution of noise to the stability
    metrics it is advantageous to first de-noise the bandpass solutions along the 
    frequency axis using a median filter with a width chosen appropriately such that
    it does not absorb genuine bandpass structure. This is particularly important for
    high frequency resolution bandpass solutions. The filter never crosses a flagged gap,
    and mirrors the data at the edge of each run of unflagged channels. Phase is smoothed
    without unwrapping (the overall delay slope is removed first and added back
    afterwards), so a short half-turn jump cannot shift the rest of the band by a turn.
    Two optional cleaning steps can be run before the smoothing; both write to the FLAG
    column of the output table and are described under
    [Radio frequency interference](#radio-frequency-interference).
2. `bandpass-map`: The bandpass table only contains antenna indices and there is a risk
    that these could change between successive bandpass calibration scans should a different 
    sub array be used or an antenna becomes unavailable. This script will construct a CSV file
    that contains a mapping of antenna indices to physical antennas for the set of bandpass
    calibration tables under scrutiny. There is a hardwired assumption that the name of each 
    bandpass table contains within it the <i>complete</i> directory name of the parent MS,
    and that the MS sits in the same directory as the table. Everything up to and including
    the first `.ms` in the table name is taken as the MS name, with any leading `cal_` prefix
    removed, so `cal_observation.ms.B0` (or `cal_observation.ms_smooth.B0`) maps to
    `observation.ms`.
3. `bandpass-lag`: Finally, this script ingests the CSV antenna mapping as well as the (smoothed)
    bandpass tables. Each pair of time-adjacent tables is compared, and plots of per-pair stability
    metrics are produced for each antenna as a function of frequency and time, as well as a CSV
    dump of the relevant stats.

Each tool has a minimal CLI which can be viewed using the `-h` command line switch.

The outputs are written for each antenna into `lag_residual_results/<antenna_name>/`. Each subdirectory contains:

- `<antenna>_pair_metrics.csv`: One row per adjacent table pair, with the
  amplitude and phase residual statistics described below, and the channel counts
  used and clipped.
- `<antenna>_epoch_metrics.csv`: One row per table (epoch), with the number of valid
  channels and the number flagged by the time-direction clip.
- `<antenna>_pair_rms.png`: The per-pair RMS values plotted as a function of time, along
  with the mean-removed (and, for phase, delay-removed) versions. A sudden jump in this
  plot indicates a sudden change in the instrumental bandpass over that interval.
- `<antenna>_channel_summary.png`: The median amplitude and phase residual (16th / 84th percentile range)
  per channel, calculated across all adjacent
  pairs. This plot shows which channels are most/least stable over time, as well as any overall frequency 
  dependence in the bandpass stability.


## Metrics

### Per-pair stability metric

For each adjacent pair of bandpass tables ${B_n}(\nu)$ and ${B_{n+1}}(\nu)$ the lagged complex ratio $R_n(\nu)$ is formed:

$$
R_n(\nu) = \frac{B_{n+1}(\nu)}{B_{n}(\nu)}
$$

The log-amplitude and phase residuals are then:

$$
A_n(\nu) = \log|R_n(\nu)|, \qquad
\qquad \phi_n(\nu) = \arg\big(R_n(\nu)\big).
$$

The phase is taken on the branch nearest its own (circular) centre over the band, so a residual
sitting close to $\pm180^\circ$ is not split in two by wrapping.

The RMS of $A_n(\nu)$ and $\phi_n(\nu)$ are then determined over all common channels for a given pair:

$$
\mathrm{RMS}_{A,n} = \sqrt{\big\langle A_n(\nu)^2 \big\rangle_\nu}, \qquad
\mathrm{RMS}_{\phi,n} = \sqrt{\big\langle \phi_n(\nu)^2 \big\rangle_\nu}.
$$

These values are written to the per-antenna `pair_metrics.csv` files under the 
`amplitude_rms_percent_approx` and `phase_rms_degrees` columns, as well as plotted
in the `<antenna>_pair_rms.png` as a function of time / table pair.

Because these are RMS values about zero, they include any overall offset between the two epochs
as well as changes in the bandpass *shape*. To separate the two, the CSV also contains:

- `amplitude_mean_percent_approx` and `phase_mean_degrees`: the mean of $A_n(\nu)$ and $\phi_n(\nu)$,
  i.e. a flat gain or phase shift between the epochs;
- `amplitude_std_percent_approx` and `phase_std_degrees`: the standard deviation about that mean,
  i.e. the change in shape alone ($\mathrm{RMS}^2 = \mathrm{mean}^2 + \mathrm{std}^2$);
- `delay_ns`: the slope of a straight line fitted to $\phi_n(\nu)$, expressed as a delay (the fit is
  robust to slopes that wrap several times across the band; the column is empty if the table has no
  `SPECTRAL_WINDOW` subtable to give the channel width);
- `phase_delay_removed_rms_degrees`: the phase RMS after removing that fitted offset and slope.

### Per-channel overview

Across all adjacent pairs for one antenna (i.e. over the pair index $n$), the median and 16th/84th percentile spread of $A_n(\nu)$ and $\phi_n(\nu)$ is determined, independently for each channel:

$$
\tilde{A}(\nu) = \mathrm{median}_{\,n}\big[A_n(\nu)\big], \qquad
\tilde{\phi}(\nu) = \mathrm{median}_{\,n}\big[\phi_n(\nu)\big]
$$

These quantities are plotted on the two panels of `<antenna>_channel_summary.png`, with the solid line being the median and the shaded region the 16th–84th percentile range.

### Interpreting the numbers

Each table takes part in two adjacent pairs (the one before it and the one after), so neighbouring
values in `pair_metrics.csv` are not independent: a single bad epoch shows up as two consecutive
high values rather than one. Also, a ratio of two noisy bandpasses carries the noise of both, so for
uncorrelated noise of equal size in each epoch the noise floor of $A_n$ and $\phi_n$ is about
$\sqrt{2}$ times the per-epoch noise.


## Radio frequency interference

Strong (residual) radio interference present in the visibilities will also corrupt the corresponding
bandpass solutions. This will then egregiously affect the stability metrics, introducing
strong deviations from epoch to epoch that are unrelated to the stability of the actual telescope.
This will generally affect a limited region of the band and is thus easily identifiable in the per-channel
overview plots, however despite that it can still adversely affect the per-pair stability metrics,
giving the impression of significant epoch to epoch variability. 

While it is assumed that radio frequency interference (RFI) has been flagged as standard prior to
the creation of the input set of bandpass tables, experience shows that strong residuals can remain,
particularly close to regions of high RFI occupancy such as those caused by the geolocation 
satellites in L-band. In practice these residuals often sit right at the edges of flagged regions,
in the wings of RFI that was flagged but not quite widely enough.

In an attempt to guard against this, outlier channels are identified at three points. All of them
clip against a *local* baseline, a running median along frequency, rather than one level for the
whole band, so that smooth genuine structure (band edges, ripple, slopes) is followed by the baseline
and kept. Phase is handled on the circle throughout, so half-turn jumps and wrapping cannot distort
the baseline. Each clip iterates, recomputing the baseline from the surviving channels, until an
iteration occurs where zero channels are dropped.

1. **Single tables, in `bandpass-smooth` (optional).** Before smoothing, `--flag-grow N` widens every
   flagged region by `N` channels on each side, and `--clip-sigma S` flags channels whose log-amplitude
   or phase deviates from the running median by more than `S` times the local robust sigma (a running
   median absolute deviation, so the noisier band edges are judged on their own terms). Both are off
   by default; `--clip-sigma 5` is a reasonable starting point, and `--flag-grow` is worth trying if
   residuals cluster at flag boundaries. Newly flagged channels are written to the FLAG column of the
   smoothed table, keep their original values, are counted in the log, and are marked with crosses in
   the diagnostic plots.
2. **Across epochs, in `bandpass-lag`.** For each antenna, every epoch is compared with the
   per-channel median of all epochs (after removing that epoch's own phase offset and delay), which
   cancels the static bandpass shape. Each epoch's deviation is then clipped against a wide running
   median (`4 x --clip-window` channels), with the noise scale at each channel measured from how much
   that channel varies across epochs. This catches features that no single-table method can separate
   from real structure, such as a block of several tens of channels flipped by $180^\circ$ in one
   epoch. Because the baseline is wide, a genuine change in one epoch that spans more than roughly a
   quarter of that width (about 100 channels by default) is followed by the baseline and kept: it is
   exactly the kind of instability these tools are meant to measure. This step needs at least three
   epochs and is controlled with `--time-clip-sigma` (default 5, 0 disables). The counts are in
   `<antenna>_epoch_metrics.csv`.
3. **Per pair, in `bandpass-lag`.** $A_n(\nu)$ and $\phi_n(\nu)$ (the latter after removing its own
   delay slope) are clipped against a running median of `--clip-window` channels (default 95), with
   the noise scale at each channel measured across all pairs for that antenna. Threshold:
   `--clip-sigma` (default 5, 0 disables). In stages 2 and 3 the running median spans flagged gaps
   (using only good channels), since ratios and deviations are roughly continuous across them, so a
   feature next to a gap is judged against data on both sides.

Finally, any channel clipped in more than a fraction `--mask-fraction` (default 0.25) of the pairs in
which it was valid is excluded from *every* pair, so that the per-pair numbers are computed over a
common set of channels and a jump in the time series cannot simply reflect a change in which channels
were used. Setting it to 1 disables this.

The `<antenna>_pair_metrics.csv` file contains `amplitude_channels_clipped` and `phase_channels_clipped` columns, 
which are the counts of channels removed by the per-pair clip for a given pair, and
`amplitude_channels_used` and `phase_channels_used`, the number of channels that actually went into the
statistics. It is advisable to examine these numbers (along with the general flag statistics of the
observation that was used to generate the bandpass table) prior to drawing any conclusions. When the
input tables have been smoothed, the noise is correlated over many channels, and an occasional run of
a few tens of channels can be clipped near the edge of a flagged gap even in clean data; this has a
negligible effect on the statistics but does show up in the counts.


## Reference antennas 

Bandpass solutions are generally solved for with respect to a chosen 'reference
antenna' to break degeneracies in the process. The phases for this antenna are fixed to zero, 
and all other phase corrections are given with respect to it. The most obvious upshot of this
is that for the chosen reference antenna the phase stability will simply be consistent with floating
point noise, at around $10^{-7}$ degrees.

The second and less obvious effect is that a sudden change in the phase stability of the reference
antenna will propagate coherently into all of the other antennas in the array. Thus if the set of
`<antenna>_pair_rms.png` images exhibit a sudden jump that is common to all antennas the prime
suspect is likely to be a sudden change in the behaviour of the reference antenna. The same applies
to features in individual channels: a block of channels that jumps in phase in every antenna except
the reference antenna points to the reference antenna.

In order to produce stability metrics for the reference antenna itself a set of bandpass tables
that utilise an alternative (known to be good) reference antenna should be produced. These will 
of course be subject to the same set of caveats.

There is also a possible scenario whereby a set of bandpass solutions are being produced for a
given antenna which unexpectedly fails or is compromised for some of the input scans. When this
happens some software will switch to a different antenna and continue the process. This generally
has no effect on the quality of the calibration, simply introducing a different zero-point for 
the relative phases at a certain time. It does however introduce discontinuities that will
adversely affect a stability analysis such as the one deployed here. Users should ensure that the
set of bandpass tables that are fed to these tools have a consistently good reference antenna
throughout.

Note that amplitude stability metrics are not affected by reference antenna issues.
